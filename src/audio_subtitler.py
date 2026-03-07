import io
import json

import stt2vtt
import webvtt
from faster_whisper import WhisperModel
from typing import BinaryIO, Union, List, Dict, Literal
import numpy as np

SubtitleFormat = Literal["vtt", "srt", "json"]


def _vtt_to_srt(vtt_content: str) -> str:
    """Convert WebVTT string to SRT using webvtt-py."""
    doc = webvtt.WebVTT.from_string(vtt_content)
    buf = io.StringIO()
    doc.write(buf, format="srt")
    return buf.getvalue()


class AudioSubtitler:
    def __init__(self, **kwargs):
        self.model = WhisperModel(**kwargs)
    
    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        format: SubtitleFormat = "vtt",
        **kwargs
    ) -> str:
        kwargs.setdefault("word_timestamps", True)
        kwargs.setdefault("vad_parameters", {"min_silence_duration_ms": 500})
        
        segments, info = self.model.transcribe(audio=audio, **kwargs)
        segments_list = self._segments_to_stt_json(segments)

        # For JSON format, return detailed Whisper segments as JSON string
        if format == "json":
            return json.dumps(segments_list, ensure_ascii=False)

        # VTT via stt2vtt (fast-whisper segments -> WebVTT)
        vtt_content = stt2vtt(segments_list)

        if format == "vtt":
            return vtt_content
        # SRT: convert VTT to SRT using webvtt-py
        return _vtt_to_srt(vtt_content)

    def _segments_to_stt_json(self, segments) -> List[Dict]:
        """Build fast-whisper style segment list for stt2vtt (id, start, end, text, words)."""
        out = []
        for i, segment in enumerate(segments):
            out.append({
                "id": getattr(segment, "id", i),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word}
                    for w in segment.words
                    if getattr(w, "word", None) is not None
                ],
            })
        return out
