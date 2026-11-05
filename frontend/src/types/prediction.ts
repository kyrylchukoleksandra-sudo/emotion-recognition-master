// Імовірності для кожної з 7 базових емоцій
export interface EmotionProbabilities {
  angry: number;
  disgust: number;
  fear: number;
  happy: number;
  sad: number;
  surprise: number;
  neutral: number;
}

// Відповідь ендпоінту /api/v1/predict
export interface EmotionPredictionResponse {
  dominant_emotion: keyof EmotionProbabilities;
  confidence: number;
  probabilities: EmotionProbabilities;
  valence: number; // Емоційна валентність (-1.0 до 1.0)
  arousal: number; // Емоційне збудження (0.0 до 1.0)
  bounding_box: [number, number, number, number]; // [x, y, width, height]
  fused_vector_dim?: number;
  execution_provider?: string;
}

// Структура координат 478 ключових точок обличчя (MediaPipe Face Mesh / Dlib)
export interface LandmarkPoint {
  x: number;
  y: number;
  z?: number;
}

// Відповідь ендпоінту /api/v1/landmarks
export interface LandmarkResponse {
  landmarks_count: number;
  landmarks: LandmarkPoint[];
  bounding_box: [number, number, number, number];
}

// Один запис знайденого схожого обличчя з бази pgvector
export interface SearchResultRecord {
  record_id: number;
  image_name: string;
  emotion_label: string;
  similarity_score: number; // Косинусна схожість
  image_url?: string;
}

// Відповідь ендпоінту /api/v1/search
export interface SimilaritySearchResponse {
  top_k: number;
  results: SearchResultRecord[];
}

// Структура запису історії аналізу для аналітичних графіків та звітів
export interface AnalyticsFrameData {
  timestamp: string;
  dominant_emotion: string;
  confidence: number;
  valence: number;
  arousal: number;
  probabilities: EmotionProbabilities;
}