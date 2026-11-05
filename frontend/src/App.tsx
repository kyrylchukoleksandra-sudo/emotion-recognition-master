import React, { useState } from 'react';
import { WebcamFeed } from './components/WebcamFeed';
import { EmotionPanel } from './components/EmotionPanel';
import { ResultsGrid } from './components/ResultsGrid';
import { PreviewModal } from './components/PreviewModal';
import { ReportExport } from './components/ReportExport';
import { AuthBar, UserProfile } from './components/AuthBar';
import { PrivacyConsentToggle } from './components/PrivacyConsentToggle';
import { ImageUpload } from './components/ImageUpload';
import { VideoUpload } from './components/VideoUpload';
import {
  EmotionPredictionResponse,
  SearchResultRecord,
  AnalyticsFrameData,
} from './types/prediction';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'webcam' | 'images' | 'video'>('webcam');
  const [currentPrediction, setCurrentPrediction] =
    useState<EmotionPredictionResponse | null>(null);
  const [history, setHistory] = useState<AnalyticsFrameData[]>([]);
  const [searchResults] = useState<SearchResultRecord[]>([]);
  const [selectedRecord, setSelectedRecord] =
    useState<SearchResultRecord | null>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  // Authentication & Privacy Consent States
  const [user, setUser] = useState<UserProfile | null>(null);
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('token')
  );
  const [isPublicConsent, setIsPublicConsent] = useState<boolean>(false);

  const handleLoginSuccess = (userProfile: UserProfile, accessToken: string) => {
    setUser(userProfile);
    setToken(accessToken);
    localStorage.setItem('token', accessToken);
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
  };

  const handlePredictionUpdate = (
    prediction: EmotionPredictionResponse | null
  ) => {
    setCurrentPrediction(prediction);

    if (prediction) {
      const newFrame: AnalyticsFrameData = {
        timestamp: new Date().toLocaleTimeString(),
        dominant_emotion: prediction.dominant_emotion,
        confidence: prediction.confidence,
        valence: prediction.valence,
        arousal: prediction.arousal,
        probabilities: prediction.probabilities,
      };

      setHistory((prev) => [...prev.slice(-29), newFrame]);
    }
  };

  const handleSelectRecord = (record: SearchResultRecord) => {
    setSelectedRecord(record);
    setIsModalOpen(true);
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: '#fbf7f5',
        color: '#4a403a',
        padding: '32px',
        boxSizing: 'border-box',
      }}
    >
      {/* Header */}
      <header
        style={{
          borderBottom: '1px solid #f0e6e1',
          paddingBottom: '20px',
          marginBottom: '24px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div>
          <h1 style={{ margin: 0, fontSize: '24px', color: '#8c7f78', fontWeight: 700 }}>
            Facial Emotion Recognition ✨
          </h1>
          <p style={{ margin: '6px 0 0 0', fontSize: '13px', color: '#b0a39c' }}>
            Master Thesis Dashboard — Real-time Analysis & pgvector Similarity
          </p>
        </div>

        <AuthBar
          user={user}
          onLoginSuccess={handleLoginSuccess}
          onLogout={handleLogout}
        />
      </header>

      {/* Standalone Privacy Consent Toggle */}
      <PrivacyConsentToggle
        isPublicConsent={isPublicConsent}
        setIsPublicConsent={setIsPublicConsent}
      />

      {/* Navigation Tabs */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '28px' }}>
        <button
          onClick={() => setActiveTab('webcam')}
          style={{
            padding: '10px 20px',
            borderRadius: '12px',
            border: 'none',
            backgroundColor: activeTab === 'webcam' ? '#b279a7' : '#FFFFFF',
            color: activeTab === 'webcam' ? '#FFFFFF' : '#6e625a',
            fontWeight: 'bold',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          }}
        >
          📷 Live Webcam Stream
        </button>
        <button
          onClick={() => setActiveTab('images')}
          style={{
            padding: '10px 20px',
            borderRadius: '12px',
            border: 'none',
            backgroundColor: activeTab === 'images' ? '#b279a7' : '#FFFFFF',
            color: activeTab === 'images' ? '#FFFFFF' : '#6e625a',
            fontWeight: 'bold',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          }}
        >
          🖼️ Batch Image Upload
        </button>
        <button
          onClick={() => setActiveTab('video')}
          style={{
            padding: '10px 20px',
            borderRadius: '12px',
            border: 'none',
            backgroundColor: activeTab === 'video' ? '#b279a7' : '#FFFFFF',
            color: activeTab === 'video' ? '#FFFFFF' : '#6e625a',
            fontWeight: 'bold',
            cursor: 'pointer',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
          }}
        >
          🎥 Video File Analysis
        </button>
      </div>

      {/* Main Grid */}
      <main
        style={{
          display: 'grid',
          gridTemplateColumns: activeTab === 'webcam' ? '1fr 360px' : '1fr', // 👈 Якщо не веб-камера, розтягуємо на всю ширину або прибираємо бічну панель
          gap: '32px',
          alignItems: 'start',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {activeTab === 'webcam' && (
            <WebcamFeed 
              onPredictionUpdate={handlePredictionUpdate} 
              isPublicConsent={isPublicConsent}
            />
          )}

          {activeTab === 'images' && (
            <ImageUpload isPublicConsent={isPublicConsent} />
          )}

          {activeTab === 'video' && (
            <VideoUpload isPublicConsent={isPublicConsent} />
          )}

          <ResultsGrid
            results={searchResults}
            onSelectRecord={handleSelectRecord}
          />
        </div>

        {/* Права колонка відображається ТІЛЬКИ для вкладки веб-камери */}
        {activeTab === 'webcam' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <EmotionPanel prediction={currentPrediction} />
            <ReportExport
              currentPrediction={currentPrediction}
              history={history}
            />
          </div>
        )}
      </main>

      <PreviewModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        prediction={currentPrediction}
        selectedRecord={selectedRecord}
      />
    </div>
  );
};

export default App;