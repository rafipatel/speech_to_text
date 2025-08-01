import gradio as gr
import whisperx
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv("hf_token")

# Model config
device = "cpu"
batch_size = 16
compute_type = "int8"

# Load main model
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

title = "🎙️ Multilingual Audio Processor"
description = "Upload an audio file and select whether to transcribe, align words, or identify speakers (Powered by WhisperX)."

def clean_alignment(result):
    cleaned_segments = []
    for seg in result.get("segments", []):
        cleaned_words = []
        for word in seg.get("words", []):
            cleaned_words.append({
                "word": word["word"],
                "start": float(word["start"]),
                "end": float(word["end"]),
                "score": float(word["score"])
            })
        cleaned_segments.append({
            "text": seg["text"],
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "words": cleaned_words
        })
    return {"segments": cleaned_segments}

def process_audio(audio_path, transcribe=True, align=False, diarize=False):
    transcript_output = ""
    align_output = {}
    diarize_output = ""

    audio = whisperx.load_audio(audio_path)
    result = None

    # Step 1: Transcribe
    # if transcribe:
    result = model.transcribe(audio, batch_size=batch_size)
    transcript_output = " ".join(seg["text"] for seg in result["segments"])

    # Step 2: Align
    if align and result:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
        align_output = clean_alignment(result)

    # Step 3: Diarization
    if diarize and result:
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        diarize_output = [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg["text"]
            } for seg in result["segments"]
            ]

    return transcript_output , align_output or {}, diarize_output or "No diarization."

with gr.Blocks(title=title, theme=gr.themes.Default(), analytics_enabled=True) as demo:
    gr.Markdown(f"<h1 style='text-align: center;font-size: 40px;'>{title}</h1>")
    gr.Markdown(f"<p style='text-align: center; font-size: 16px;'>{description}</p>")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            transcribe_checkbox = gr.Markdown("✅ Transcription will always be performed.")
            align_checkbox = gr.Checkbox(label="Align")
            diarize_checkbox = gr.Checkbox(label="Diarize")
            gr.Markdown("### <span style='font-size: 18px;'>🎧 Try Sample Audio</span>")
            gr.Examples(
                examples=[[f"test_audios/{audio_file}"] for audio_file in os.listdir("test_audios") if audio_file.endswith(('.mp3', '.wav'))],
                inputs=[audio_input],
                label=""
            )
        with gr.Column(scale=2):
            transcript_output = gr.Textbox(label="📄 Transcript", lines=10, interactive=False)
            alignment_output = gr.JSON(label="🧭 Word Alignment")
            diarization_output = gr.JSON(label="🗣️ Speaker Diarization")
    with gr.Row():
        process_button = gr.Button("Process")

    process_button.click(
        fn=process_audio,
        inputs=[audio_input, transcribe_checkbox, align_checkbox, diarize_checkbox],
        outputs=[transcript_output, alignment_output, diarization_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)