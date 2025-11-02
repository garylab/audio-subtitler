# Audio Subtitler

Convert audio files to subtitles (VTT, SRT) using Faster-Whisper.

[![PyPI](https://img.shields.io/pypi/v/audio-subtitler.svg)](https://pypi.org/project/audio-subtitler/)
[![Python Versions](https://img.shields.io/pypi/pyversions/audio-subtitler.svg)](https://pypi.org/project/audio-subtitler/)
[![Downloads](https://static.pepy.tech/badge/audio-subtitler)](https://pepy.tech/project/audio-subtitler)
[![Monthly Downloads](https://img.shields.io/pypi/dm/audio-subtitler.svg)](https://pypi.org/project/audio-subtitler/)
[![Run on RunPod](https://img.shields.io/badge/Run%20on-RunPod-6b3cff?logo=runpod&logoColor=white)](https://runpod.io?ref=hh0mhml0)

## Features

- 🚀 **Full Faster-Whisper support** - All features and parameters from faster-whisper
- 📝 **Multiple formats** - VTT (WebVTT) and SRT subtitle output
- 🎯 **Smart auto-detection** - Automatically detects format from file extension
- 🌍 **Multi-language** - Supports 100+ languages with auto-detection
- ⚡ **GPU acceleration** - CUDA support for faster transcription
- 🎙️ **Voice Activity Detection** - Automatically removes silence
- 💻 **Simple APIs** - Easy-to-use CLI and Python API
- 🐳 **Docker GPU support** - Ready for serverless deployment

## Installation

```bash
pip install audio-subtitler
```

Optional dependencies:
```bash
pip install audio-subtitler[runpod]  # For RunPod serverless
pip install audio-subtitler[dev]     # For development
```

## Quick Start

### CLI

```bash
# Auto-detect format from file extension (recommended)
audiosubtitler input.mp3 -o output.vtt
audiosubtitler input.mp3 -o output.srt

# Specify options
audiosubtitler input.mp3 -o output.vtt --model large-v3 --language en --device cuda

# Output to stdout
audiosubtitler input.mp3 --format srt > output.srt

# Use shorter command
audiosub input.mp3 -o output.vtt
```

### Python API

```python
from src import AudioSubtitler

# Initialize
converter = AudioSubtitler(
    model_size_or_path="base",
    device="cpu",
    compute_type="int8"
)

# Transcribe
result = converter.transcribe("audio.mp3", format="vtt", language="en")

# Access results
print(result["content"])     # Subtitle content
print(result["format"])      # "vtt" or "srt"
print(result["word_count"])  # Number of words
```

## API Reference

### AudioSubtitler

**Constructor**: `AudioSubtitler(**kwargs)`

Accepts all [faster-whisper WhisperModel](https://github.com/SYSTRAN/faster-whisper) parameters:
- `model_size_or_path`: Model name (tiny, base, small, medium, large, large-v3) or path
- `device`: "cpu", "cuda", or "auto"
- `compute_type`: "int8", "int8_float16", "int16", "float16", "float32"
- `cpu_threads`, `num_workers`, `download_root`, `local_files_only`, etc.

**Method**: `transcribe(audio, format="vtt", **kwargs)`

Parameters:
- `audio`: File path (str), file object (BinaryIO), or numpy array
- `format`: "vtt" or "srt" (default: "vtt")
- `**kwargs`: All [faster-whisper transcribe](https://github.com/SYSTRAN/faster-whisper#transcribe) parameters
  - `language`, `beam_size`, `vad_filter`, `vad_parameters`, `word_timestamps`, etc.

Returns:
```python
{
    "content": str,      # Subtitle content
    "format": str,       # "vtt" or "srt"
    "word_count": int    # Word count
}
```

## Docker (GPU only)

```bash
docker-compose -f docker-compose-gpu.yml up
```

Input/Output for RunPod serverless:
```json
// Input
{
  "input": {
    "audio": "<base64_encoded_audio>",
    "language": "en",
    "format": "vtt"
  }
}

// Output
{
  "content": "WEBVTT\n\n00:00:00.000 --> ...",
  "format": "vtt",
  "word_count": 150
}
```

## Output Examples

**VTT:**
```
WEBVTT

00:00:00.000 --> 00:00:03.500
Hello, this is a test transcription.

00:00:03.500 --> 00:00:07.200
The audio is converted to text with timestamps.
```

**SRT:**
```
1
00:00:00,000 --> 00:00:03,500
Hello, this is a test transcription.

2
00:00:03,500 --> 00:00:07,200
The audio is converted to text with timestamps.
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size |
| `WHISPER_DEVICE` | `cpu` | cpu, cuda, auto |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute type |
| `WHISPER_BEAM_SIZE` | `5` | Beam size |


## License

MIT License - see [LICENSE](LICENSE) file for details.
