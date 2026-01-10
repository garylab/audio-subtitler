import runpod
import io
import os
import base64

# Lazy imports to speed up startup
subtitler = None
MODEL_SIZE_OR_PATH = "large-v3"
DOWNLOAD_ROOT = "models"


def download_model():
    from faster_whisper import WhisperModel
    WhisperModel(MODEL_SIZE_OR_PATH, device="cpu", download_root=DOWNLOAD_ROOT)


def get_subtitler():
    global subtitler
    if subtitler is None:
        from src.audio_subtitler import AudioSubtitler
        subtitler = AudioSubtitler(
            model_size_or_path=MODEL_SIZE_OR_PATH,
            device="cuda",
            device_index=[int(i) for i in os.getenv("WHISPER_DEVICE_INDEX", "0").split(",")],
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
            num_workers=int(os.getenv("WHISPER_NUM_WORKERS", "1")),
            download_root=DOWNLOAD_ROOT,
            local_files_only=True,
        )
    return subtitler


def handler(event):
    """Handler for RunPod serverless."""
    job_input = event.get("input", {})
    
    # Health check / ping - return immediately (no model loading)
    if not job_input or job_input.get("ping"):
        return {"status": "ok", "message": "ready"}
    
    audio_base64 = job_input.get("audio")
    if not audio_base64:
        return {"status": "error", "message": "No audio provided"}
    
    format = job_input.get("format", "vtt")
    
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception as e:
        return {"status": "error", "message": f"Invalid base64: {e}"}

    try:
        transcribe_kwargs = {
            "format": format,
            "beam_size": int(os.getenv("WHISPER_BEAM_SIZE", "5")),
            "vad_filter": True,
        }
        
        if format == "json":
            transcribe_kwargs["suppress_tokens"] = []
            transcribe_kwargs["condition_on_previous_text"] = False
            transcribe_kwargs["vad_filter"] = False
        
        result = get_subtitler().transcribe(io.BytesIO(audio_data), **transcribe_kwargs)
        return {"status": "ok", "output": result}
    except Exception as e:
        return {"status": "error", "message": f"Transcription failed: {e}"}


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
