import React, { useRef, useState, useEffect } from 'react';

interface VideoUploadProps {
  isPublicConsent: boolean;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  isPublicConsent,
}) => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [trackingMode, setTrackingMode] = useState<'multi' | 'single'>('multi');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [videoData, setVideoData] = useState<any | null>(null);
  
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setVideoData(null);
    }
  };

  const handleVideoSubmit = async () => {
    if (!videoFile) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('mode', trackingMode);
    formData.append('is_public', String(isPublicConsent));

    try {
      const response = await fetch('http://localhost:8000/api/v1/video', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setVideoData(data);
      }
    } catch (err) {
      console.error('Failed to process video:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  // Експорт звіту у форматі JSON
  const handleExportJSON = () => {
    if (!videoData) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(videoData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `video_analytics_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Експорт звіту у форматі CSV
  const handleExportCSV = () => {
    if (!videoData || !videoData.timeline) return;

    const headers = ['Timestamp(s)', 'Frame', 'Dominant Emotion', 'Confidence(%)', 'Valence', 'Arousal'];
    const rows = videoData.timeline.map((item: any) => [
      item.timestamp_sec,
      item.frame_number,
      item.dominant_emotion,
      (item.confidence * 100).toFixed(1),
      item.valence !== undefined && item.valence !== null ? Number(item.valence).toFixed(4) : '',
      item.arousal !== undefined && item.arousal !== null ? Number(item.arousal).toFixed(4) : '',
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e: any[]) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `video_summary_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  // Малювання рамки поверх відео
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !videoData || !videoData.timeline) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const activeFrame = videoData.timeline.find(
      (item: any) => Math.abs(item.timestamp_sec - currentTime) < 0.5
    );

    canvas.width = video.videoWidth || video.clientWidth;
    canvas.height = video.videoHeight || video.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (activeFrame && activeFrame.bounding_box && activeFrame.bounding_box.length === 4) {
      const [x, y, w, h] = activeFrame.bounding_box;

      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = Math.max(3, canvas.width / 200);
      ctx.strokeRect(x, y, w, h);

      const confidencePct = activeFrame.confidence ? (activeFrame.confidence * 100).toFixed(0) : '95';
      const label = `${activeFrame.dominant_emotion.toUpperCase()} (${confidencePct}%)`;
      
      ctx.font = 'bold 16px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = '#22c55e';
      ctx.fillRect(x, y - 28 > 0 ? y - 28 : y, textWidth + 16, 26);

      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(label, x + 8, (y - 28 > 0 ? y - 28 : y) + 18);
    }
  }, [currentTime, videoData]);

  const currentActiveFrame = videoData?.timeline?.find(
    (item: any) => Math.abs(item.timestamp_sec - currentTime) < 0.5
  );

  return (
    <div
      style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '20px',
        padding: '24px',
        boxShadow: '0 4px 20px rgba(180, 160, 150, 0.08)',
        border: '1px solid #f0e6e1',
        display: 'grid',
        gridTemplateColumns: videoData ? '1.5fr 1fr' : '1fr',
        gap: '24px',
      }}
    >
      {/* Ліва частина: плеєр та елементи керування */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <h3 style={{ margin: 0, color: '#6e625a', fontSize: '18px' }}>
          Interactive Video Emotion Tracker 🎥
        </h3>

        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ fontSize: '13px', color: '#6e625a' }}
        />

        {videoUrl && (
          <div style={{ position: 'relative', width: '100%', lineHeight: 0, backgroundColor: '#000', borderRadius: '12px', overflow: 'hidden' }}>
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              style={{ width: '100%', display: 'block', height: 'auto' }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                objectFit: 'contain',
              }}
            />
          </div>
        )}

        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <label style={{ fontSize: '13px', color: '#6e625a', cursor: 'pointer' }}>
            <input
              type="radio"
              name="mode"
              checked={trackingMode === 'multi'}
              onChange={() => setTrackingMode('multi')}
              style={{ marginRight: '6px', accentColor: '#b279a7' }}
            />
            Multi-Face Tracking
          </label>
          <label style={{ fontSize: '13px', color: '#6e625a', cursor: 'pointer' }}>
            <input
              type="radio"
              name="mode"
              checked={trackingMode === 'single'}
              onChange={() => setTrackingMode('single')}
              style={{ marginRight: '6px', accentColor: '#b279a7' }}
            />
            Single Target Focus
          </label>
        </div>

        <button
          onClick={handleVideoSubmit}
          disabled={isProcessing || !videoFile}
          style={{
            padding: '12px',
            backgroundColor: isProcessing ? '#e0d6d1' : '#b279a7',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: '12px',
            fontWeight: 'bold',
            cursor: isProcessing ? 'not-allowed' : 'pointer',
          }}
        >
          {isProcessing ? 'Processing Video & Bounding Boxes...' : 'Run Video Analysis'}
        </button>
      </div>

      {/* Права частина: Метрики, підсумок та Експорт звіту */}
      {videoData && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', maxHeight: '650px', overflowY: 'auto' }}>
          
          {/* Блок генерації та експорту звітів (стиль із дизайну) */}
          <div
            style={{
              backgroundColor: '#1e1e1e',
              borderRadius: '16px',
              padding: '18px 20px',
              boxShadow: '0 8px 24px rgba(0,0,0,0.15)',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}
          >
            <h4 style={{ margin: 0, color: '#FFFFFF', fontSize: '15px', fontWeight: 'bold' }}>
              Export Analytics & Reports 📊
            </h4>
            <div style={{ display: 'flex', gap: '10px' }}>
              <button
                onClick={handleExportJSON}
                style={{
                  flex: 1,
                  padding: '10px 14px',
                  backgroundColor: '#1d4ed8',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '10px',
                  fontWeight: 'bold',
                  fontSize: '13px',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s ease',
                }}
              >
                Export JSON
              </button>
              <button
                onClick={handleExportCSV}
                style={{
                  flex: 1,
                  padding: '10px 14px',
                  backgroundColor: '#15803d',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '10px',
                  fontWeight: 'bold',
                  fontSize: '13px',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s ease',
                }}
              >
                Export CSV Summary
              </button>
            </div>
          </div>

          {/* Live Frame Metrics */}
          <div style={{ padding: '16px', backgroundColor: '#fcf8f6', borderRadius: '16px', border: '1px solid #ebdcd5' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '15px', color: '#6e625a' }}>
              Live Frame Metrics 📊
            </h4>
            <div style={{ fontSize: '13px', color: '#a0938a', marginBottom: '10px' }}>
              Timestamp: <strong style={{ color: '#b279a7' }}>{currentTime.toFixed(2)}s</strong>
            </div>

            {currentActiveFrame ? (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <div>
                    <span style={{ fontSize: '11px', color: '#a0938a', display: 'block' }}>Dominant</span>
                    <strong style={{ fontSize: '18px', color: '#b279a7', textTransform: 'capitalize' }}>
                      {currentActiveFrame.dominant_emotion}
                    </strong>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '11px', color: '#a0938a', display: 'block' }}>Confidence</span>
                    <strong style={{ fontSize: '16px', color: '#6e625a' }}>
                      {((currentActiveFrame.confidence || 0) * 100).toFixed(1)}%
                    </strong>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '14px' }}>
                  <div style={{ padding: '8px', backgroundColor: '#faf6f4', borderRadius: '8px', border: '1px solid #f0e6e1' }}>
                    <div style={{ fontSize: '10px', color: '#a0938a' }}>Valence</div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#6e625a' }}>
                      {currentActiveFrame.valence !== undefined && currentActiveFrame.valence !== null
                        ? Number(currentActiveFrame.valence).toFixed(3)
                        : 'N/A'}
                    </div>
                  </div>
                  <div style={{ padding: '8px', backgroundColor: '#faf6f4', borderRadius: '8px', border: '1px solid #f0e6e1' }}>
                    <div style={{ fontSize: '10px', color: '#a0938a' }}>Arousal</div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#6e625a' }}>
                      {currentActiveFrame.arousal !== undefined && currentActiveFrame.arousal !== null
                        ? Number(currentActiveFrame.arousal).toFixed(3)
                        : 'N/A'}
                    </div>
                  </div>
                </div>

                {currentActiveFrame.probabilities && (
                  <div>
                    <div style={{ fontSize: '12px', color: '#6e625a', fontWeight: 'bold', marginBottom: '6px' }}>
                      Probabilities
                    </div>
                    {Object.entries(currentActiveFrame.probabilities).map(([emotion, prob]: [string, any]) => (
                      <div key={emotion} style={{ marginBottom: '6px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#6e625a', marginBottom: '2px' }}>
                          <span style={{ textTransform: 'capitalize' }}>{emotion}</span>
                          <span>{(prob * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ height: '5px', backgroundColor: '#f0e6e1', borderRadius: '3px', overflow: 'hidden' }}>
                          <div
                            style={{
                              width: `${prob * 100}%`,
                              height: '100%',
                              backgroundColor: emotion.toLowerCase() === currentActiveFrame.dominant_emotion?.toLowerCase() ? '#b279a7' : '#d2b4c8',
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: '12px', color: '#a0938a', fontStyle: 'italic', marginTop: '10px' }}>
                No face detected at this timestamp.
              </div>
            )}
          </div>

          {/* Video Summary */}
          <div style={{ padding: '16px', backgroundColor: '#fcf8f6', borderRadius: '16px', border: '1px solid #ebdcd5' }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '15px', color: '#6e625a' }}>Video Summary</h4>
            <p style={{ fontSize: '13px', margin: '4px 0', color: '#6e625a' }}>
              <strong>Dominant:</strong> {videoData.summary.dominant_emotion}
            </p>
            <p style={{ fontSize: '13px', margin: '4px 0', color: '#6e625a' }}>
              <strong>Total Frames:</strong> {videoData.summary.total_frames_analyzed}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};