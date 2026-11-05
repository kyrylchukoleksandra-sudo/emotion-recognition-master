import React from 'react';
import { EmotionPredictionResponse, SearchResultRecord } from '../types/prediction';

interface PreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  prediction?: EmotionPredictionResponse | null;
  selectedRecord?: SearchResultRecord | null;
}

export const PreviewModal: React.FC<PreviewModalProps> = ({
  isOpen,
  onClose,
  prediction,
  selectedRecord,
}) => {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    >
      <div
        style={{
          backgroundColor: '#1E1E1E',
          borderRadius: '12px',
          padding: '24px',
          width: '500px',
          maxWidth: '90%',
          color: '#FFF',
          boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5)',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderBottom: '1px solid #333',
            paddingBottom: '12px',
          }}
        >
          <h3 style={{ margin: 0, fontSize: '18px' }}>Frame Inspection Details</h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#AAA',
              fontSize: '20px',
              cursor: 'pointer',
            }}
          >
            ✕
          </button>
        </div>

        {selectedRecord && (
          <div>
            <h4 style={{ color: '#00E5FF', margin: '0 0 8px 0' }}>pgvector Match Record</h4>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Record ID:</strong> {selectedRecord.record_id}
            </p>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Image Name:</strong> {selectedRecord.image_name}
            </p>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Emotion Label:</strong> {selectedRecord.emotion_label}
            </p>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Similarity Score:</strong> {(selectedRecord.similarity_score * 100).toFixed(2)}%
            </p>
          </div>
        )}

        {prediction && (
          <div>
            <h4 style={{ color: '#28A745', margin: '0 0 8px 0' }}>Current Prediction Specs</h4>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Dominant Emotion:</strong> {prediction.dominant_emotion.toUpperCase()}
            </p>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%
            </p>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Fused Vector Dimension:</strong> {prediction.fused_vector_dim || 512}D
            </p>
            <p style={{ margin: '4px 0', fontSize: '14px', color: '#DDD' }}>
              <strong>Execution Provider:</strong> {prediction.execution_provider || 'CPUExecutionProvider'}
            </p>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '12px' }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              backgroundColor: '#0066FF',
              color: '#FFF',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};