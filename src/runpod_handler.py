import runpod
import io
import os
import base64
from faster_whisper import WhisperModel

from src.audio_subtitler import AudioSubtitler


subtitler = None
MODEL_SIZE_OR_PATH = "large-v3"
DOWNLOAD_ROOT = "models"


def download_model():
    WhisperModel(MODEL_SIZE_OR_PATH, device="cpu", download_root=DOWNLOAD_ROOT)


def get_subtitler():
    global subtitler
    if subtitler is None:
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
    job_input = event.get("input", {})
    
    # Health check / ping - return immediately
    if job_input.get("ping"):
        return {"status": "ok", "model": MODEL_SIZE_OR_PATH}
    
    audio_base64 = job_input.get("audio")
    format = job_input.get("format", "vtt")
    
    if not audio_base64:
        return {"status": "error", "message": "No audio data provided. Please provide 'audio' field with base64 encoded audio data."}
    
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception as e:
        return {"status": "error", "message": f"Failed to decode base64 audio data: {str(e)}"}

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
        return {"status": "error", "message": f"Transcription failed: {str(e)}"}


if __name__ == '__main__':
    # Start handler immediately - model loads on first request
    # Pre-loading can cause timeout during RunPod's init test
    print(f"Starting handler (model: {MODEL_SIZE_OR_PATH}, will load on first request)...")
    runpod.serverless.start({'handler': handler})
