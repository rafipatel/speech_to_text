# FastAPI WhisperX Transcription Service

This repository provides a FastAPI-based API for audio transcription, alignment, and speaker diarization using [WhisperX](https://github.com/m-bain/whisperx).

👉 Try the live demo on [Hugging Face Spaces](https://huggingface.co/spaces/Rafii/SpeechSegmenter)

---

## 📁 Project Structure

```
fastapi/
├── app/
│   └── main.py                # FastAPI app with /transcribe endpoint
├── test_audios/               # Example audio files for testing
│   ├── BernhardtCrescent.wav
│   ├── BlackStone_en_in.mp4
│   ├── BlackStone_en_in.wav
│   ├── fillicafe.wav
│   └── harvard.wav
├── requirements.txt           # Python dependencies
├── dockerfile                 # Docker setup
```

---

## ⚙️ Setup & Local Development

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd fastapi
```

### 2. Create and Activate a Virtual Environment

### Option A: Python
```sh
python3 -m venv .venv
source .venv/bin/activate
```

### Option B: Conda (Recommended)

Build and run the container:

```sh
conda create --name whisperx_api python==3.10 
conda activate whisperx_api
```

---

### 3. Install Python Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install System Dependencies

- **ffmpeg** is required for audio processing.
- On Ubuntu/Debian:
  ```sh
  sudo apt-get update && sudo apt-get install -y ffmpeg git
  ```

---

## 🚀 Running the API

### Option A: Local (Recommended for development)

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B: Docker

Build and run the container:

```sh
docker build -t curify_fastapi .  
docker run -p 8000:8000 curify_fastapi
```

---

## 🧪 Testing the API

### 1. Using `curl`

Upload an audio file from `test_audios/` for transcription:

```sh
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audios/harvard.wav"
```

### 2. Using Swagger UI

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

---

## 📝 Endpoint Overview

- **POST `/transcribe`**  
  Upload an audio file and receive a transcript with speaker labels and timestamps.

---

## To extract audio from video using ffmpeg

```sh
pip install ffmpeg
ffmpeg -i BlackStone_en_in.mp4 -ar 16000 -ac 1 BlackStone_en_in.wav
```

## 🧹 Cleanup

Temporary files are automatically deleted after each request.

---


## 🛠️ Notes

- The WhisperX model is loaded once at startup for efficiency.
- Diarization uses a Hugging Face token (edit in `main.py` if needed).
- For best results, use clear audio files (see `test_audios/` for examples).

---
