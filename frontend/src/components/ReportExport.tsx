import React from 'react';
import { EmotionPredictionResponse, AnalyticsFrameData } from '../types/prediction';

interface ReportExportProps {
  currentPrediction: EmotionPredictionResponse | null;
  history: AnalyticsFrameData[];
}

export const ReportExport: React.FC<ReportExportProps> = ({
  currentPrediction,
  history,
}) => {
  // Download session analytics as JSON
  const exportToJSON = () => {
    const reportData = {
      export_timestamp: new Date().toISOString(),
      current_prediction: currentPrediction,
      session_history: history,
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: 'application/json',
    });
    downloadFile(blob, `emotion_analytics_report_${Date.now()}.json`);
  };

  // Download session analytics as CSV
  const exportToCSV = () => {
    if (history.length === 0) {
      alert('No history data available to export.');
      return;
    }

    const headers = [
      'Timestamp',
      'Dominant Emotion',
      'Confidence (%)',
      'Valence',
      'Arousal',
      'Happy',
      'Surprise',
      'Neutral',
      'Sad',
      'Angry',
      'Fear',
      'Disgust',
    ];

    const rows = history.map((item) => [
      item.timestamp,
      item.dominant_emotion,
      (item.confidence * 100).toFixed(2),
      item.valence.toFixed(2),
      item.arousal.toFixed(2),
      (item.probabilities.happy * 100).toFixed(2),
      (item.probabilities.surprise * 100).toFixed(2),
      (item.probabilities.neutral * 100).toFixed(2),
      (item.probabilities.sad * 100).toFixed(2),
      (item.probabilities.angry * 100).toFixed(2),
      (item.probabilities.fear * 100).toFixed(2),
      (item.probabilities.disgust * 100).toFixed(2),
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `emotion_session_report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Helper function to trigger browser download
  const downloadFile = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        backgroundColor: '#1E1E1E',
        padding: '20px',
        borderRadius: '12px',
        color: '#FFF',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)',
      }}
    >
      <div style={{ fontSize: '15px', fontWeight: 'bold' }}>
        Export Analytics & Reports
      </div>

      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={exportToJSON}
          disabled={!currentPrediction && history.length === 0}
          style={{
            padding: '8px 16px',
            backgroundColor: '#0066FF',
            color: '#FFF',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '13px',
            opacity: !currentPrediction && history.length === 0 ? 0.5 : 1,
          }}
        >
          Export JSON
        </button>

        <button
          onClick={exportToCSV}
          disabled={history.length === 0}
          style={{
            padding: '8px 16px',
            backgroundColor: '#28A745',
            color: '#FFF',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '13px',
            opacity: history.length === 0 ? 0.5 : 1,
          }}
        >
          Export CSV Summary
        </button>
      </div>
    </div>
  );
};