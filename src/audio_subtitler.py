from faster_whisper import WhisperModel
from typing import BinaryIO, Union, Tuple, List, Dict, Any, Literal
import numpy as np

STOP_CHARS = set(
    ".!?,:;…‥"
    "。！？，、；："
    "।"
    "܀።፧"
    "؟؛"
    "၊။"
    "⸮⁇⁈⁉"
)

SubtitleFormat = Literal["vtt", "srt", "json"]


class AudioSubtitler:
    def __init__(self, **kwargs):
        self.model = WhisperModel(**kwargs)
    
    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        format: SubtitleFormat = "vtt",
        **kwargs
    ) -> Dict[str, Any]:
        kwargs.setdefault("word_timestamps", True)
        kwargs.setdefault("vad_filter", True)
        kwargs.setdefault("vad_parameters", {"min_silence_duration_ms": 500})
        
        segments, _ = self.model.transcribe(audio=audio, **kwargs)
        
        # For JSON format, return original Whisper segments directly
        if format == "json":
            segments_list = []
            word_count = 0
            for segment in segments:
                word_count += len(segment.words) if segment.words else 0
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                        {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                        for w in (segment.words or [])
                    ],
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                })
            return {
                "content": segments_list,
                "format": format,
                "word_count": word_count,
            }
        
        subtitles, word_count = self.segments_to_subtitle(segments)
        content = self._format_subtitles(subtitles, format)
        
        return {
            "content": content,
            "format": format,
            "word_count": word_count
        }
    
    def _format_subtitles(self, subtitles: List[Dict], format: SubtitleFormat) -> str:
        items = []
        
        for idx, subtitle in enumerate(subtitles, start=1):
            text = subtitle.get("msg")
            if not text:
                continue
            
            start_time = subtitle.get("start_time")
            end_time = subtitle.get("end_time")
            
            if format == "vtt":
                items.append(self._format_vtt_segment(text, start_time, end_time))
            else:
                items.append(self._format_srt_segment(idx, text, start_time, end_time))
        
        if format == "vtt":
            return "WEBVTT\n\n" + "\n".join(items)
        else:
            return "\n".join(items)

    def segments_to_subtitle(self, segments) -> Tuple[List[Dict], int]:
        subtitles = []
        word_count = 0
        
        for segment in segments:
            word_count += len(segment.words)
            words_idx = 0
            words_len = len(segment.words)

            seg_start = 0
            seg_end = 0
            seg_text = ""

            if segment.words:
                is_segmented = False
                for word in segment.words:
                    if not is_segmented:
                        seg_start = word.start
                        is_segmented = True

                    seg_end = word.end
                    seg_text += word.word

                    if self.end_with_stop_char(word.word):
                        seg_text = seg_text[:-1]
                        if not seg_text:
                            continue

                        if seg_start < seg_end and seg_text.strip():
                            subtitles.append(
                                {
                                    "msg": seg_text,
                                    "start_time": seg_start,
                                    "end_time": seg_end,
                                }
                            )

                        is_segmented = False
                        seg_text = ""

                    if words_idx == 0 and segment.start < word.start:
                        seg_start = word.start
                    if words_idx == (words_len - 1) and segment.end > word.end:
                        seg_end = word.end
                    words_idx += 1

            if not seg_text:
                continue

            if seg_start < seg_end and seg_text.strip():
                subtitles.append(
                    {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
                )

        return subtitles, word_count


    def _seconds_to_time(self, seconds: float, separator: str = ".") -> str:
        hours = int(seconds // 3600)
        seconds = seconds % 3600
        minutes = int(seconds // 60)
        milliseconds = int(seconds * 1000) % 1000
        seconds = int(seconds % 60)
        return "{:02d}:{:02d}:{:02d}{}{:03d}".format(
            hours, minutes, seconds, separator, milliseconds
        )
    
    def seconds_to_vtt_time(self, seconds: float) -> str:
        return self._seconds_to_time(seconds, ".")
    
    def seconds_to_srt_time(self, seconds: float) -> str:
        return self._seconds_to_time(seconds, ",")
    
    def _capitalize_text(self, text: str) -> str:
        text = text.strip()
        return text[0].upper() + text[1:] if text else text
    
    def _format_vtt_segment(self, text: str, start_time: float, end_time: float) -> str:
        start_time_str = self.seconds_to_vtt_time(start_time)
        end_time_str = self.seconds_to_vtt_time(end_time)
        text = self._capitalize_text(text)
        return f"{start_time_str} --> {end_time_str}\n{text}\n"
    
    def _format_srt_segment(self, index: int, text: str, start_time: float, end_time: float) -> str:
        start_time_str = self.seconds_to_srt_time(start_time)
        end_time_str = self.seconds_to_srt_time(end_time)
        text = self._capitalize_text(text)
        return f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n"

    def end_with_stop_char(self, text: str) -> bool:
        if not text:
            return False

        for c in STOP_CHARS:
            if text.endswith(c):
                return True
        return False

