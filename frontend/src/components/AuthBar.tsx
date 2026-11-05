import React from 'react';
import { GoogleLogin, googleLogout } from '@react-oauth/google';

export interface UserProfile {
  id: number;
  email: string;
  name: string;
  picture: string;
  role: 'user' | 'admin';
}

interface AuthBarProps {
  user: UserProfile | null;
  onLoginSuccess: (user: UserProfile, token: string) => void;
  onLogout: () => void;
}

export const AuthBar: React.FC<AuthBarProps> = ({
  user,
  onLoginSuccess,
  onLogout,
}) => {
  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/auth/google', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential: credentialResponse.credential }),
      });
      const data = await response.json();
      if (response.ok) {
        onLoginSuccess(data.user, data.access_token);
      }
    } catch (err) {
      console.error('Login Error:', err);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'flex-end',
        backgroundColor: '#FFFFFF',
        padding: '12px 24px',
        borderRadius: '16px',
        boxShadow: '0 4px 20px rgba(180, 160, 150, 0.08)',
        border: '1px solid #f0e6e1',
        marginBottom: '24px',
      }}
    >
      {/* User Status / Login */}
      <div>
        {user ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <img
              src={user.picture}
              alt={user.name}
              style={{ width: '36px', height: '36px', borderRadius: '50%' }}
            />
            <div style={{ textAlign: 'left' }}>
              <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#4a403a' }}>
                {user.name} {user.role === 'admin' && '👑 (Admin)'}
              </div>
              <div style={{ fontSize: '11px', color: '#a0938a' }}>{user.email}</div>
            </div>
            <button
              onClick={() => {
                googleLogout();
                onLogout();
              }}
              style={{
                padding: '6px 14px',
                backgroundColor: '#f7d6d0',
                color: '#c96f68',
                border: 'none',
                borderRadius: '10px',
                fontSize: '12px',
                fontWeight: 'bold',
                cursor: 'pointer',
              }}
            >
              Sign Out
            </button>
          </div>
        ) : (
          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={() => console.log('Login Failed')}
            useOneTap
          />
        )}
      </div>
    </div>
  );
};