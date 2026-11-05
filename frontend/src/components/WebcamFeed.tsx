import React, { useEffect } from 'react';
import { useCamera } from '../hooks/useCamera';

interface WebcamFeedProps {
  onPredictionUpdate?: (prediction: any) => void;
  isPublicConsent: boolean;
}

export const WebcamFeed: React.FC<WebcamFeedProps> = ({ onPredictionUpdate, isPublicConsent }) => {
  const {
    videoRef,
    canvasRef,
    isStreaming,
    isPaused,
    prediction,
    error,
    startCamera,
    stopCamera,
    togglePause,
  } = useCamera(300);

  // Передаємо передбачення батьківському компоненту, якщо потрібно
  useEffect(() => {
    if (onPredictionUpdate) {
      onPredictionUpdate(prediction);
    }
  }, [prediction, onPredictionUpdate]);

  // Малювання рамки обличчя та бляшки з емоцією на Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Очищаємо Canvas перед кожним кадром
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (prediction && prediction.bounding_box && videoRef.current) {
      const video = videoRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      const [x, y, w, h] = prediction.bounding_box;
      const confidencePercent = (prediction.confidence * 100).toFixed(0);
      const label = `${prediction.dominant_emotion.toUpperCase()} ${confidencePercent}%`;

      // 1. Зелена рамка навколо обличчя
      ctx.strokeStyle = '#00FF66';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // 2. Фон для тексту
      ctx.font = 'bold 16px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      const textHeight = 26;
      const boxY = y - textHeight > 0 ? y - textHeight : y;

      ctx.fillStyle = 'rgba(0, 255, 102, 0.85)';
      ctx.fillRect(x, boxY, textWidth + 14, textHeight);

      // 3. Текст із назвою емоції
      ctx.fillStyle = '#000000';
      ctx.fillText(label, x + 7, boxY + 18);
    }
  }, [prediction, videoRef, canvasRef]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
      <div
        style={{
          position: 'relative',
          width: '640px',
          height: '480px',
          backgroundColor: '#1e1e1e',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)',
        }}
      >
        <video
          ref={videoRef}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          playsInline
          muted
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
          }}
        />

        {!isStreaming && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: '#888',
              fontSize: '18px',
            }}
          >
            Вебкамера вимкнена
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: '12px' }}>
        {!isStreaming ? (
          <button
            onClick={startCamera}
            style={{
              padding: '10px 24px',
              borderRadius: '8px',
              backgroundColor: '#0066FF',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '15px',
            }}
          >
            Увімкнути камеру
          </button>
        ) : (
          <>
            <button
              onClick={togglePause}
              style={{
                padding: '10px 20px',
                borderRadius: '8px',
                backgroundColor: isPaused ? '#28A745' : '#FFC107',
                color: '#fff',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '15px',
              }}
            >
              {isPaused ? '▶ Відновити відео' : '⏸ Пауза (зафіксувати емоцію)'}
            </button>

            <button
              onClick={stopCamera}
              style={{
                padding: '10px 20px',
                borderRadius: '8px',
                backgroundColor: '#DC3545',
                color: '#fff',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '15px',
              }}
            >
              Вимкнути
            </button>
          </>
        )}
      </div>

      {error && <p style={{ color: '#FF4D4D', margin: 0 }}>{error}</p>}
    </div>
  );
};