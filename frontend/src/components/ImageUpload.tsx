import React, { useState, useEffect } from 'react';

interface ImageUploadProps {
  isPublicConsent: boolean;
  onAnalysisComplete?: (results: any) => void;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  isPublicConsent,
  onAnalysisComplete,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [analysisResults, setAnalysisResults] = useState<any[]>([]);
  const [modalHistory, setModalHistory] = useState<Array<{ result: any; previewUrl?: string }>>([]);

  const currentModalItem = modalHistory.length > 0 ? modalHistory[modalHistory.length - 1] : null;

  useEffect(() => {
    return () => {
      previewUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [previewUrls]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      previewUrls.forEach((url) => URL.revokeObjectURL(url));

      setSelectedFiles(e.target.files);
      const urls = Array.from(e.target.files).map((file) =>
        URL.createObjectURL(file)
      );
      setPreviewUrls(urls);
      setAnalysisResults([]);
      setModalHistory([]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    Array.from(selectedFiles).forEach((file) => {
      formData.append('files', file);
    });
    formData.append('is_public', String(isPublicConsent));

    try {
      const response = await fetch('http://localhost:8000/api/v1/batch-images', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token') || ''}`,
        },
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        const resultsArray = data.results || [];
        setAnalysisResults(resultsArray);

        if (onAnalysisComplete) {
          onAnalysisComplete(data);
        }
      }
    } catch (err) {
      console.error('Failed to process images:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleOpenModal = (result: any, previewUrl?: string) => {
    setModalHistory([{ result, previewUrl }]);
  };

  const handlePushToModal = (result: any, previewUrl?: string) => {
    setModalHistory((prev) => [...prev, { result, previewUrl }]);
  };

  const handlePopModal = () => {
    setModalHistory((prev) => prev.slice(0, prev.length - 1));
  };

  const handleCloseModal = () => {
    setModalHistory([]);
  };

  return (
    <div
      style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '20px',
        padding: '24px',
        boxShadow: '0 4px 20px rgba(180, 160, 150, 0.08)',
        border: '1px solid #f0e6e1',
      }}
    >
      <h3 style={{ margin: '0 0 12px 0', color: '#6e625a', fontSize: '18px' }}>
        Batch Image Upload 📸
      </h3>
      <p style={{ margin: '0 0 20px 0', fontSize: '13px', color: '#a0938a' }}>
        Upload single or multiple images for emotion analysis and feature indexing.
      </p>

      <input
        type="file"
        multiple
        accept="image/*"
        onChange={handleFileChange}
        style={{
          marginBottom: '16px',
          display: 'block',
          width: '100%',
          fontSize: '13px',
          color: '#6e625a',
        }}
      />

      {/* Image Preview Grid */}
      {previewUrls.length > 0 && (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
            gap: '10px',
            marginBottom: '16px',
          }}
        >
          {previewUrls.map((url, idx) => (
            <img
              key={idx}
              src={url}
              alt="preview"
              style={{
                width: '100%',
                height: '80px',
                objectFit: 'cover',
                borderRadius: '10px',
                border: '1px solid #e0d6d1',
              }}
            />
          ))}
        </div>
      )}

      <button
        onClick={handleUpload}
        disabled={isUploading || !selectedFiles}
        style={{
          width: '100%',
          padding: '10px 18px',
          backgroundColor: isUploading ? '#e0d6d1' : '#b279a7',
          color: '#FFFFFF',
          border: 'none',
          borderRadius: '12px',
          fontWeight: 'bold',
          cursor: isUploading ? 'not-allowed' : 'pointer',
          marginBottom: analysisResults.length > 0 ? '20px' : '0',
        }}
      >
        {isUploading ? 'Analyzing Images...' : 'Analyze & Store Features'}
      </button>

      {/* Results Grid */}
      {analysisResults.length > 0 && (
        <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <h4 style={{ margin: '0 0 8px 0', color: '#6e625a', fontSize: '15px' }}>
            Analysis Results ({analysisResults.length})
          </h4>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
              gap: '12px',
            }}
          >
            {analysisResults.map((res, idx) => {
              const hasError = Boolean(res.error);
              const previewUrl = previewUrls[idx] || '';
              const filename = res.filename || res.image_name || `Image #${idx + 1}`;

              return (
                <div
                  key={idx}
                  onClick={() => {
                    if (!hasError) {
                      handleOpenModal(res, previewUrl);
                    }
                  }}
                  style={{
                    backgroundColor: hasError ? '#fff5f5' : '#fcf8f6',
                    borderRadius: '14px',
                    padding: '14px',
                    border: hasError ? '1px solid #fecaca' : '1px solid #ebdcd5',
                    cursor: hasError ? 'default' : 'pointer',
                    transition: 'all 0.15s ease',
                    userSelect: 'none',
                  }}
                  onMouseEnter={(e) => {
                    if (!hasError) {
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 16px rgba(178, 121, 167, 0.12)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!hasError) {
                      e.currentTarget.style.transform = 'none';
                      e.currentTarget.style.boxShadow = 'none';
                    }
                  }}
                >
                  <div
                    style={{
                      fontSize: '12px',
                      fontWeight: 'bold',
                      color: '#8c7f78',
                      marginBottom: '6px',
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                    title={filename}
                  >
                    📄 {filename}
                  </div>

                  {hasError ? (
                    <div style={{ color: '#ef4444', fontSize: '12px' }}>
                      ⚠️ {res.error}
                    </div>
                  ) : (
                    <div>
                      <div
                        style={{
                          fontSize: '14px',
                          fontWeight: 'bold',
                          color: '#b279a7',
                          textTransform: 'capitalize',
                          marginBottom: '4px',
                        }}
                      >
                        {res.dominant_emotion || 'Unknown'}
                      </div>
                      <div style={{ fontSize: '12px', color: '#a0938a', marginBottom: '6px' }}>
                        Confidence: {((res.confidence || 0) * 100).toFixed(1)}%
                      </div>
                      <div
                        style={{
                          width: '100%',
                          height: '6px',
                          backgroundColor: '#e5dcd6',
                          borderRadius: '3px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${(res.confidence || 0) * 100}%`,
                            height: '100%',
                            backgroundColor: '#b279a7',
                          }}
                        />
                      </div>
                      <div style={{ fontSize: '11px', color: '#b279a7', marginTop: '8px', textAlign: 'right', fontWeight: 'bold' }}>
                        Click to inspect 🔍
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {currentModalItem && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.45)',
            backdropFilter: 'blur(5px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
          onClick={handleCloseModal}
        >
          <div
            style={{
              backgroundColor: '#FFFFFF',
              borderRadius: '24px',
              padding: '28px',
              maxWidth: '560px',
              width: '90%',
              maxHeight: '85vh',
              overflowY: 'auto',
              boxShadow: '0 20px 40px rgba(0,0,0,0.18)',
              border: '1px solid #f0e6e1',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header Navigation with Back Button */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {modalHistory.length > 1 && (
                  <button
                    onClick={handlePopModal}
                    style={{
                      backgroundColor: '#f4ece8',
                      border: 'none',
                      borderRadius: '8px',
                      padding: '4px 10px',
                      color: '#b279a7',
                      fontSize: '13px',
                      fontWeight: 'bold',
                      cursor: 'pointer',
                    }}
                  >
                    ← Back
                  </button>
                )}
                <h3 style={{ margin: 0, color: '#6e625a', fontSize: '18px' }}>
                  Detailed Analysis 📊
                </h3>
              </div>

              <button
                onClick={handleCloseModal}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '22px',
                  cursor: 'pointer',
                  color: '#a0938a',
                }}
              >
                ✕
              </button>
            </div>

            {/* Visual Bounding Box Overlay on Image */}
            {currentModalItem.previewUrl && (
              <div style={{ position: 'relative', marginBottom: '20px', display: 'flex', justifyContent: 'center' }}>
                <img
                  id="modal-analyzed-image"
                  src={currentModalItem.previewUrl}
                  alt="Analyzed face"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '260px',
                    borderRadius: '16px',
                    objectFit: 'contain',
                    backgroundColor: '#fcf8f6',
                    border: '1px solid #ebdcd5',
                  }}
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    const bbox = currentModalItem.result.bounding_box;
                    if (!bbox || bbox.length < 4) return;

                    const canvas = document.getElementById('bbox-canvas') as HTMLCanvasElement;
                    if (!canvas) return;

                    canvas.width = img.clientWidth;
                    canvas.height = img.clientHeight;

                    const ctx = canvas.getContext('2d');
                    if (!ctx) return;

                    const scaleX = img.clientWidth / img.naturalWidth;
                    const scaleY = img.clientHeight / img.naturalHeight;

                    const [x, y, w, h] = bbox;
                    const scaledX = x * scaleX;
                    const scaledY = y * scaleY;
                    const scaledW = w * scaleX;
                    const scaledH = h * scaleY;

                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    ctx.strokeStyle = '#b279a7';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

                    const label = `${(currentModalItem.result.dominant_emotion || '').toUpperCase()}`;
                    ctx.font = 'bold 12px Inter, sans-serif';
                    const textWidth = ctx.measureText(label).width;

                    ctx.fillStyle = '#b279a7';
                    ctx.fillRect(scaledX, scaledY - 22 > 0 ? scaledY - 22 : scaledY, textWidth + 10, 20);

                    ctx.fillStyle = '#FFFFFF';
                    ctx.fillText(label, scaledX + 5, (scaledY - 22 > 0 ? scaledY - 22 : scaledY) + 14);
                  }}
                />

                <canvas
                  id="bbox-canvas"
                  style={{
                    position: 'absolute',
                    top: 0,
                    pointerEvents: 'none',
                  }}
                />
              </div>
            )}

            {/* Dominant Emotion & Confidence Header */}
            <div
              style={{
                backgroundColor: '#fcf8f6',
                borderRadius: '16px',
                padding: '16px',
                marginBottom: '16px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                border: '1px solid #ebdcd5',
              }}
            >
              <div>
                <span style={{ fontSize: '12px', color: '#a0938a', display: 'block' }}>Dominant Emotion</span>
                <strong style={{ fontSize: '24px', color: '#b279a7', textTransform: 'capitalize' }}>
                  {currentModalItem.result.dominant_emotion}
                </strong>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '12px', color: '#a0938a', display: 'block' }}>Confidence</span>
                <strong style={{ fontSize: '24px', color: '#6e625a' }}>
                  {((currentModalItem.result.confidence || 0) * 100).toFixed(1)}%
                </strong>
              </div>
            </div>

            {/* Conditionally Display Valence & Arousal (Тільки коли дані є) */}
            {currentModalItem.result.valence !== undefined &&
              currentModalItem.result.valence !== null &&
              currentModalItem.result.arousal !== undefined &&
              currentModalItem.result.arousal !== null && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '20px' }}>
                  <div style={{ padding: '14px', backgroundColor: '#faf6f4', borderRadius: '14px', border: '1px solid #f0e6e1' }}>
                    <div style={{ fontSize: '12px', color: '#a0938a', marginBottom: '4px' }}>Valence</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#6e625a' }}>
                      {Number(currentModalItem.result.valence).toFixed(4)}
                    </div>
                  </div>
                  <div style={{ padding: '14px', backgroundColor: '#faf6f4', borderRadius: '14px', border: '1px solid #f0e6e1' }}>
                    <div style={{ fontSize: '12px', color: '#a0938a', marginBottom: '4px' }}>Arousal</div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#6e625a' }}>
                      {Number(currentModalItem.result.arousal).toFixed(4)}
                    </div>
                  </div>
                </div>
              )}

            {/* Conditionally Display Emotion Probabilities Distribution */}
            {currentModalItem.result.probabilities && Object.keys(currentModalItem.result.probabilities).length > 0 && (
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#6e625a' }}>
                  All Emotion Probabilities
                </h4>
                {Object.entries(currentModalItem.result.probabilities).map(([emotion, prob]: [string, any]) => (
                  <div key={emotion} style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#6e625a', marginBottom: '3px' }}>
                      <span style={{ textTransform: 'capitalize' }}>{emotion}</span>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '6px', backgroundColor: '#f0e6e1', borderRadius: '3px', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${prob * 100}%`,
                          height: '100%',
                          backgroundColor: emotion.toLowerCase() === currentModalItem.result.dominant_emotion?.toLowerCase() ? '#b279a7' : '#d2b4c8',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Pgvector Top-K Similar Faces */}
            {currentModalItem.result.similar_faces && currentModalItem.result.similar_faces.length > 0 && (
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ margin: '0 0 10px 0', fontSize: '14px', color: '#6e625a' }}>
                  pgvector Top Similar Profiles 🧬
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {currentModalItem.result.similar_faces.map((face: any, fIdx: number) => {
                    const targetName = face.image_name || face.filename;
                    const matchedIndex = analysisResults.findIndex(
                      (r) => (r.filename || r.image_name) === targetName
                    );

                    const matchedResult = matchedIndex !== -1 ? analysisResults[matchedIndex] : null;
                    const matchedPreview = matchedIndex !== -1 ? previewUrls[matchedIndex] : currentModalItem.previewUrl;

                    return (
                      <div
                        key={fIdx}
                        onClick={() => {
                          const displayResult = matchedResult || {
                            dominant_emotion: face.emotion_label || 'Unknown',
                            confidence: face.confidence || 1.0,
                            filename: targetName,
                            image_name: targetName,
                            bounding_box: [],
                            similar_faces: [],
                          };

                          handlePushToModal(displayResult, matchedPreview);
                        }}
                        style={{
                          padding: '12px 14px',
                          backgroundColor: '#faf6f4',
                          borderRadius: '12px',
                          border: '1px solid #ebdcd5',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          fontSize: '13px',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                          userSelect: 'none',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#f4ece8';
                          e.currentTarget.style.transform = 'translateX(4px)';
                          e.currentTarget.style.borderColor = '#b279a7';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = '#faf6f4';
                          e.currentTarget.style.transform = 'none';
                          e.currentTarget.style.borderColor = '#ebdcd5';
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          {matchedPreview ? (
                            <img
                              src={matchedPreview}
                              alt="thumb"
                              style={{ width: '28px', height: '28px', borderRadius: '6px', objectFit: 'cover' }}
                            />
                          ) : (
                            <span style={{ fontSize: '16px' }}>🖼️</span>
                          )}
                          <div>
                            <span style={{ color: '#6e625a', fontWeight: '600', display: 'block' }}>
                              {targetName || `Face #${face.id}`}
                            </span>
                            {face.emotion_label && (
                              <span style={{ fontSize: '11px', color: '#a0938a', textTransform: 'capitalize' }}>
                                Emotion: {face.emotion_label}
                              </span>
                            )}
                          </div>
                        </div>

                        <div style={{ textAlign: 'right' }}>
                          <span style={{ color: '#b279a7', fontWeight: 'bold', display: 'block' }}>
                            Distance: {Number(face.distance).toFixed(4)}
                          </span>
                          <span style={{ fontSize: '10px', color: '#b279a7', fontWeight: 'bold' }}>
                            Inspect Match 🔍
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <button
              onClick={handleCloseModal}
              style={{
                width: '100%',
                padding: '12px',
                backgroundColor: '#b279a7',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '12px',
                fontWeight: 'bold',
                cursor: 'pointer',
              }}
            >
              Close Details
            </button>
          </div>
        </div>
      )}
    </div>
  );
};