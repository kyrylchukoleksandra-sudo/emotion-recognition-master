import {
  EmotionPredictionResponse,
  LandmarkResponse,
  SimilaritySearchResponse,
} from '../types/prediction';

const API_BASE_URL = 'http://localhost:8000/api/v1';

/**
 * Надсилає зображення або стоп-кадр на бекенд для розпізнавання емоцій
 */
export async function predictEmotion(
  imageBlob: Blob
): Promise<EmotionPredictionResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'frame.jpg');

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || 'Помилка під час розпізнавання емоцій'
    );
  }

  return response.json();
}

/**
 * Отримує координати ключових точок обличчя (landmarks)
 */
export async function getLandmarks(imageBlob: Blob): Promise<LandmarkResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'frame.jpg');

  const response = await fetch(`${API_BASE_URL}/landmarks`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || 'Помилка під час отримання ключових точок обличчя'
    );
  }

  return response.json();
}

/**
 * Виконує векторний пошук схожих облич у базі даних pgvector
 */
export async function searchSimilarFaces(
  imageBlob: Blob,
  topK: number = 5
): Promise<SimilaritySearchResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'search.jpg');

  const response = await fetch(`${API_BASE_URL}/search?top_k=${topK}`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(
      errorData.detail || 'Помилка під час пошуку схожих облич у pgvector'
    );
  }

  return response.json();
}