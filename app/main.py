from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import whisperx
import tempfile
import os
from dotenv import load_dotenv

app = FastAPI()

device = "cpu"
batch_size = 16
compute_type = "int8"

load_dotenv()
hf_token = os.getenv("hf_token")
print(hf_token)
# Load model once during startup
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    # Save uploaded file to temp location
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Load audio
        audio = whisperx.load_audio(tmp_path)

        # Step 1: Transcribe
        result = model.transcribe(audio, batch_size=batch_size)

        # Step 2: Align
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        # Step 3: Diarization
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Format response (from curify repo)
        transcript_with_speakers = [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment.get("speaker", "SPEAKER_00")
            }
            for segment in result["segments"]
        ]

        return JSONResponse(content=transcript_with_speakers)

    finally:
        # Clean up temp file
        os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=6000)