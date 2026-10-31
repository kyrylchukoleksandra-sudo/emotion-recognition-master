
# Backend Service & ML Inference Pipeline

This directory contains the production-ready FastAPI backend service and machine learning pipeline for real-time facial emotion recognition and vector similarity search.

---

## 🛠 Features & Architecture

* **Framework**: FastAPI (Pydantic v2, SQLAlchemy 2.0)
* **ML Inference**: Dual-mode execution engine (ONNX Runtime / PyTorch)
* **Facial Landmarks & Alignment**: MediaPipe Tasks API (478 3D landmarks + 52 Blendshapes)
* **Preprocessing**: Contrast Limited Adaptive Histogram Equalization (CLAHE) & Aspect-Ratio Preserving Letterboxing
* **Multimodal Feature Fusion**: Visual Embedding (512-d) + Geometric/Mimic Feature Vector (52-d) $\rightarrow$ Fused Representation (564-d)
* **Vector Persistence & Search**: PostgreSQL with `pgvector` HNSW indexing for cosine similarity search

---

## 📋 Requirements

* Python 3.10+
* Virtual Environment (`myenv`)
* PostgreSQL with `pgvector` extension (optional for pure API prediction, required for `/save_features` and `/search`)

---

## ⚙️ Installation & Environment Setup

1. **Navigate to project root directory**:
```bash
   cd project_sem2

```

2. **Create and activate virtual environment**:
```bash
python3 -m venv myenv
source myenv/bin/activate  # On macOS/Linux
# myenv\Scripts\activate   # On Windows

```


3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install fastapi uvicorn python-multipart sqlalchemy pgvector opencv-python mediapipe numpy pillow torch onnxruntime httpx

```



---

## 🧠 Model Artifact Loading

The inference orchestrator (`EmotionPredictor`) automatically resolves model artifacts upon startup:

* **Default Location**: `artifacts/models/classifier.onnx`
* **Fallback Behavior**: If no exported `.onnx` or `.pth` weight file is detected in `artifacts/models/`, the engine seamlessly defaults to a baseline visual embedding pipeline without interrupting API availability.
* **Hot-Swapping Weights**: To connect a newly trained neural network, simply place your exported `classifier.onnx` file inside `artifacts/models/`. The backend detects and uses the ONNX Runtime execution provider automatically.

---

## 🚀 Running the Local Server

Start the development server using `uvicorn`:

```bash
./myenv/bin/python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

```

Once running, access the interactive API documentation:

* **Swagger UI**: [http://127.0.0.1:8000/docs](https://www.google.com/search?q=http://127.0.0.1:8000/docs)
* **ReDoc**: [http://127.0.0.1:8000/redoc](https://www.google.com/search?q=http://127.0.0.1:8000/redoc)
* **Health Check**: [http://127.0.0.1:8000/api/v1/health](https://www.google.com/search?q=http://127.0.0.1:8000/api/v1/health)

---

## 🛣 API Endpoints Reference

All endpoints strictly adhere to Section 3.11 of the Master's Thesis specifications:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Service health status check |
| `POST` | `/api/v1/upload` | Validates uploaded client image files |
| `POST` | `/api/v1/preprocess` | Applies face detection, normalization, and CLAHE |
| `POST` | `/api/v1/landmarks` | Returns 478 3D landmarks, bounding boxes, and FACS blendshapes |
| `POST` | `/api/v1/features` | Extracts geometric/mimic metrics and builds fused 564-d vector |
| `POST` | `/api/v1/predict` | Executes end-to-end emotion prediction (class probabilities, valence, arousal) |
| `POST` | `/api/v1/save_features` | Persists emotion labels and fused feature vectors into PostgreSQL (`pgvector`) |
| `POST` | `/api/v1/search` | Performs Top-K cosine similarity search against stored facial vector profiles |

---

## 🧪 Testing

Run unit and integration test suites:

```bash
# Run unit backend tests
./myenv/bin/python -m unittest discover -s tests/unit -v

# Run API integration tests only
./myenv/bin/python -m unittest discover -s tests/integration -v

```

