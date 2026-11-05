import React from 'react';

interface PrivacyConsentToggleProps {
  isPublicConsent: boolean;
  setIsPublicConsent: (val: boolean) => void;
}

export const PrivacyConsentToggle: React.FC<PrivacyConsentToggleProps> = ({
  isPublicConsent,
  setIsPublicConsent,
}) => {
  return (
    <div
      style={{
        backgroundColor: '#FFFFFF',
        padding: '14px 20px',
        borderRadius: '16px',
        boxShadow: '0 4px 20px rgba(180, 160, 150, 0.08)',
        border: '1px solid #f0e6e1',
        marginBottom: '16px',
        display: 'flex',
        alignItems: 'center',
      }}
    >
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          fontSize: '13px',
          color: '#6e625a',
          cursor: 'pointer',
          fontWeight: 500,
        }}
      >
        <input
          type="checkbox"
          checked={isPublicConsent}
          onChange={(e) => setIsPublicConsent(e.target.checked)}
          style={{ accentColor: '#b279a7', width: '16px', height: '16px' }}
        />
        <span>Allow adding my snapshots to shared pgvector database (Public)</span>
      </label>
    </div>
  );
};