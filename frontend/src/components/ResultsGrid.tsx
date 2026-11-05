import React from 'react';
import { SearchResultRecord } from '../types/prediction';

interface ResultsGridProps {
  results: SearchResultRecord[];
  loading?: boolean;
  onSelectRecord?: (record: SearchResultRecord) => void;
}

export const ResultsGrid: React.FC<ResultsGridProps> = ({
  results,
  loading = false,
  onSelectRecord,
}) => {
  if (loading) {
    return (
      <div
        style={{
          padding: '20px',
          color: '#AAA',
          textAlign: 'center',
          backgroundColor: '#1E1E1E',
          borderRadius: '12px',
        }}
      >
        Searching for similar facial expressions in pgvector...
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div
        style={{
          padding: '20px',
          color: '#666',
          textAlign: 'center',
          backgroundColor: '#1E1E1E',
          borderRadius: '12px',
        }}
      >
        No similar facial records found.
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        backgroundColor: '#1E1E1E',
        padding: '20px',
        borderRadius: '12px',
        width: '100%',
        maxWidth: '640px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)',
      }}
    >
      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#FFF' }}>
        Similar Faces (pgvector Nearest Neighbors)
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))',
          gap: '12px',
        }}
      >
        {results.map((item) => {
          const similarityPercent = (item.similarity_score * 100).toFixed(1);

          return (
            <div
              key={item.record_id}
              onClick={() => onSelectRecord && onSelectRecord(item)}
              style={{
                backgroundColor: '#2A2A2A',
                borderRadius: '8px',
                padding: '10px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                cursor: onSelectRecord ? 'pointer' : 'default',
                transition: 'transform 0.2s ease, background-color 0.2s ease',
                border: '1px solid #333',
              }}
            >
              <div
                style={{
                  width: '80px',
                  height: '80px',
                  backgroundColor: '#3A3A3A',
                  borderRadius: '6px',
                  overflow: 'hidden',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '8px',
                }}
              >
                {item.image_url ? (
                  <img
                    src={item.image_url}
                    alt={item.image_name}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                ) : (
                  <span style={{ fontSize: '24px', color: '#888' }}>👤</span>
                )}
              </div>

              <div
                style={{
                  fontSize: '12px',
                  fontWeight: 'bold',
                  color: '#00E5FF',
                  textTransform: 'uppercase',
                  textAlign: 'center',
                }}
              >
                {item.emotion_label}
              </div>

              <div style={{ fontSize: '11px', color: '#AAA', marginTop: '2px' }}>
                Score: {similarityPercent}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};