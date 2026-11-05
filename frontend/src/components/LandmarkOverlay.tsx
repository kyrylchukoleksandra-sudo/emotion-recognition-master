import React, { useEffect, useRef, useState } from 'react';
import { getLandmarks } from '../api/client';
import { LandmarkPoint } from '../types/prediction';

interface LandmarkOverlayProps {
  imageBlob: Blob | null;
  width?: number;
  height?: number;
}

export const LandmarkOverlay: React.FC<LandmarkOverlayProps> = ({
  imageBlob,
  width = 640,
  height = 480,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [landmarksCount, setLandmarksCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!imageBlob) return;

    const fetchAndDrawLandmarks = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await getLandmarks(imageBlob);
        setLandmarksCount(response.landmarks_count);

        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = width;
        canvas.height = height;
        ctx.clearRect(0, 0, width, height);

        // Draw facial landmark points
        ctx.fillStyle = '#00E5FF'; // Cyan color for facial mesh points
        response.landmarks.forEach((point: LandmarkPoint) => {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
          ctx.fill();
        });

        // Draw bounding box if present
        if (response.bounding_box) {
          const [x, y, w, h] = response.bounding_box;
          ctx.strokeStyle = '#00E5FF';
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]); // Dashed bounding box
          ctx.strokeRect(x, y, w, h);
          ctx.setLineDash([]);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to extract facial landmarks');
      } finally {
        setLoading(false);
      }
    };

    fetchAndDrawLandmarks();
  }, [imageBlob, width, height]);

  return (
    <div style={{ position: 'relative', width: `${width}px`, height: `${height}px` }}>
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
      {loading && (
        <div style={{ color: '#00E5FF', fontSize: '12px', marginTop: '4px' }}>
          Extracting facial landmarks...
        </div>
      )}
      {error && <div style={{ color: '#FF4D4D', fontSize: '12px' }}>{error}</div>}
      {landmarksCount > 0 && !loading && (
        <div style={{ color: '#AAA', fontSize: '12px', marginTop: '4px' }}>
          Landmarks detected: {landmarksCount} points
        </div>
      )}
    </div>
  );
};