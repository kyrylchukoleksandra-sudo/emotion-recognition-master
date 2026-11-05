import { useState, useRef, useCallback, useEffect } from 'react';
import { predictEmotion } from '../api/client';
import { EmotionPredictionResponse, AnalyticsFrameData } from '../types/prediction';

export function useCamera(fpsIntervalMs: number = 300) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<EmotionPredictionResponse | null>(null);
  const [history, setHistory] = useState<AnalyticsFrameData[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Запуск потоку з вебкамери
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setIsStreaming(true);
        setIsPaused(false);
        setError(null);
      }
    } catch (err) {
      setError('Не вдалося отримати доступ до вебкамери. Перевірте дозволи браузера.');
    }
  };

  // Зупинка вебкамери
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsPaused(false);
    setPrediction(null);
  };

  // Перемикання станом паузи (фіксація поточного кадру)
  const togglePause = () => {
    if (!videoRef.current) return;

    if (isPaused) {
      videoRef.current.play();
      setIsPaused(false);
    } else {
      videoRef.current.pause();
      setIsPaused(true);
    }
  };

  // Функція обробки та відправки одного кадру
  const processFrame = useCallback(async () => {
    if (!videoRef.current || isPaused || !isStreaming) return;

    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    // Створюємо тимчасовий прихований canvas для отримання Blob
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);

    tempCanvas.toBlob(
      async (blob) => {
        if (!blob) return;

        try {
          const result = await predictEmotion(blob);
          setPrediction(result);
          setError(null);

          // Додаємо запис до історії для побудови графіків аналітики
          const newFrame: AnalyticsFrameData = {
            timestamp: new Date().toLocaleTimeString(),
            dominant_emotion: result.dominant_emotion,
            confidence: result.confidence,
            valence: result.valence,
            arousal: result.arousal,
            probabilities: result.probabilities,
          };

          setHistory((prev) => [...prev.slice(-30), newFrame]); // зберігаємо останні 30 кадрів
        } catch (err) {
          // Якщо обличчя не виявлено
          setPrediction(null);
        }
      },
      'image/jpeg',
      0.85
    );
  }, [isPaused, isStreaming]);

  // Інтервал відправки кадрів
  useEffect(() => {
    if (!isStreaming || isPaused) return;

    const interval = setInterval(() => {
      processFrame();
    }, fpsIntervalMs);

    return () => clearInterval(interval);
  }, [isStreaming, isPaused, processFrame, fpsIntervalMs]);

  return {
    videoRef,
    canvasRef,
    isStreaming,
    isPaused,
    prediction,
    history,
    error,
    startCamera,
    stopCamera,
    togglePause,
  };
}