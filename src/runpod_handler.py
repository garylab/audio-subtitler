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
    audio_base64 = job_input.get("audio")
    format = job_input.get("format", "vtt")
    if not audio_base64:
        return {"error": "No audio data provided. Please provide 'audio' field with base64 encoded audio data."}
    
    try:
        audio_data = base64.b64decode(audio_base64)
    except Exception as e:
        return {"error": f"Failed to decode base64 audio data: {str(e)}"}

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
        
        return get_subtitler().transcribe(io.BytesIO(audio_data), **transcribe_kwargs)
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}


if __name__ == '__main__':
    # Pre-load model at startup so it's ready for RunPod health checks
    print("Loading model...")
    get_subtitler()
    print("Model loaded, starting handler...")
    runpod.serverless.start({'handler': handler})
