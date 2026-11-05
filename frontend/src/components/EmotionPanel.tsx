import React from 'react';
import { EmotionPredictionResponse, EmotionProbabilities } from '../types/prediction';

interface EmotionPanelProps {
  prediction: EmotionPredictionResponse | null;
}

// Пастельні кольори для кожної емоції
const EMOTION_CONFIG: Record<
  keyof EmotionProbabilities,
  { label: string; color: string; bg: string }
> = {
  happy: { label: 'Happy ✨', color: '#689f76', bg: '#d4e7d4' },      /* М'ята */
  surprise: { label: 'Surprise 🌸', color: '#b279a7', bg: '#e8d7f1' }, /* Лаванда */
  neutral: { label: 'Neutral ☁️', color: '#8c7f78', bg: '#f0e6e1' },  /* Крем */
  sad: { label: 'Sad 🌧️', color: '#5b8a99', bg: '#cce3de' },         /* Пастельно-блакитний */
  angry: { label: 'Angry 🍓', color: '#c96f68', bg: '#f7d6d0' },       /* Пастельно-рожевий */
  fear: { label: 'Fear 🌙', color: '#8a7090', bg: '#dfd3e3' },        /* Пастельно-бузковий */
  disgust: { label: 'Disgust 🍑', color: '#d18b53', bg: '#fce1cb' },   /* Персик */
};

export const EmotionPanel: React.FC<EmotionPanelProps> = ({ prediction }) => {
  if (!prediction) {
    return (
      <div
        style={{
          width: '360px',
          padding: '24px',
          backgroundColor: '#FFFFFF',
          borderRadius: '20px',
          color: '#8c7f78',
          textAlign: 'center',
          boxShadow: '0 8px 24px rgba(180, 160, 150, 0.08)',
          border: '1px solid #f0e6e1',
        }}
      >
        Awaiting webcam stream... ✨
      </div>
    );
  }

  const { dominant_emotion, confidence, probabilities, valence, arousal } = prediction;
  const dominantConfig = EMOTION_CONFIG[dominant_emotion] || {
    label: dominant_emotion,
    color: '#8c7f78',
    bg: '#f0e6e1',
  };

  return (
    <div
      style={{
        width: '360px',
        padding: '24px',
        backgroundColor: '#FFFFFF',
        borderRadius: '20px',
        color: '#4a403a',
        boxShadow: '0 10px 30px rgba(180, 160, 150, 0.1)',
        border: '1px solid #f0e6e1',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
      }}
    >
      {/* 1. Головна емоція */}
      <div style={{ textAlign: 'center' }}>
        <span style={{ fontSize: '12px', color: '#a0938a', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>
          Dominant Emotion
        </span>
        <div
          style={{
            fontSize: '24px',
            fontWeight: '700',
            color: dominantConfig.color,
            marginTop: '6px',
            display: 'inline-block',
            padding: '4px 16px',
            backgroundColor: dominantConfig.bg,
            borderRadius: '16px',
          }}
        >
          {dominantConfig.label}
        </div>
        <div style={{ fontSize: '13px', color: '#8c7f78', marginTop: '6px' }}>
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
      </div>

      <hr style={{ borderColor: '#f5eeea', margin: 0 }} />

      {/* 2. Когнітивні шкали */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#6e625a' }}>
          Cognitive Specs
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#8c7f78' }}>
            <span>Valence</span>
            <span>{valence.toFixed(2)}</span>
          </div>
          <div style={{ width: '100%', height: '10px', backgroundColor: '#f7f2ef', borderRadius: '10px', marginTop: '4px', overflow: 'hidden' }}>
            <div
              style={{
                width: `${((valence + 1) / 2) * 100}%`,
                height: '100%',
                backgroundColor: valence >= 0 ? '#b5d8b5' : '#f2b6b1',
                borderRadius: '10px',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#8c7f78' }}>
            <span>Arousal</span>
            <span>{arousal.toFixed(2)}</span>
          </div>
          <div style={{ width: '100%', height: '10px', backgroundColor: '#f7f2ef', borderRadius: '10px', marginTop: '4px', overflow: 'hidden' }}>
            <div
              style={{
                width: `${arousal * 100}%`,
                height: '100%',
                backgroundColor: '#fce1cb',
                borderRadius: '10px',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        </div>
      </div>

      <hr style={{ borderColor: '#f5eeea', margin: 0 }} />

      {/* 3. Прогрес-бари 7 емоцій */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#6e625a' }}>
          Probabilities
        </div>

        {(Object.keys(probabilities) as Array<keyof EmotionProbabilities>).map((key) => {
          const value = probabilities[key];
          const percent = (value * 100).toFixed(1);
          const config = EMOTION_CONFIG[key];

          return (
            <div key={key}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '12px',
                  color: key === dominant_emotion ? '#4a403a' : '#9e9189',
                  fontWeight: key === dominant_emotion ? 'bold' : 'normal',
                }}
              >
                <span>{config.label}</span>
                <span>{percent}%</span>
              </div>
              <div
                style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: '#f7f2ef',
                  borderRadius: '10px',
                  marginTop: '4px',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    width: `${percent}%`,
                    height: '100%',
                    backgroundColor: config.bg,
                    borderRadius: '10px',
                    transition: 'width 0.3s ease',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};