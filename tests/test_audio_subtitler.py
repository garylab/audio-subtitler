import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from src.audio_subtitler import AudioSubtitler


class MockWord:
    """Mock object for Whisper word timestamps"""
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class MockSegment:
    """Mock object for Whisper segment"""
    def __init__(self, text: str, start: float, end: float, words: list):
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class TestAudioSubtitler:
    """Test suite for AudioSubtitler class"""

    @patch('src.audio_subtitler.WhisperModel')
    def test_init_default_params(self, mock_whisper_model):
        """Test initialization with default parameters"""
        converter = AudioSubtitler(model_size_or_path="base")
        mock_whisper_model.assert_called_once_with(model_size_or_path="base")
        assert converter.model is not None

    @patch('src.audio_subtitler.WhisperModel')
    def test_init_custom_params(self, mock_whisper_model):
        """Test initialization with custom parameters"""
        converter = AudioSubtitler(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8"
        )
        mock_whisper_model.assert_called_once_with(
            model_size_or_path="base",
            device="cpu",
            compute_type="int8"
        )

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_default_params(self, mock_whisper_model):
        """Test transcribe with default parameters"""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3")
        
        # Verify default parameters were set
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is True
        assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}
        
        # transcribe() returns VTT string by default
        assert isinstance(result, str)
        assert result.startswith("WEBVTT\n\n")

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_custom_params(self, mock_whisper_model):
        """Test transcribe with custom parameters"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe(
            "test.mp3",
            language="en",
            beam_size=10,
            temperature=0.5
        )
        
        # Verify custom parameters were passed
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["beam_size"] == 10
        assert call_kwargs["temperature"] == 0.5

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_override_defaults(self, mock_whisper_model):
        """Test that defaults can be overridden"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe(
            "test.mp3",
            word_timestamps=False,
            vad_filter=False,
            vad_parameters={"min_silence_duration_ms": 1000}
        )
        
        call_kwargs = mock_model_instance.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is False
        assert call_kwargs["vad_filter"] is False
        assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 1000}

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_vtt_format(self, mock_whisper_model):
        """Test that VTT output is correctly formatted"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3")
        
        # transcribe() returns VTT string (via stt2vtt)
        assert result.startswith("WEBVTT\n\n")
        assert "00:00:00.000 --> 00:00:01.000" in result
        assert "Hello world" in result

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_with_numpy_array(self, mock_whisper_model):
        """Test transcribe with numpy array input"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        audio_array = np.zeros(16000, dtype=np.float32)
        result = converter.transcribe(audio_array)
        
        assert isinstance(result, str)
        assert result.startswith("WEBVTT\n\n")

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_multiple_segments(self, mock_whisper_model):
        """Test transcribe with multiple segments"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        segment1_words = [MockWord("First", 0.0, 0.5), MockWord(" sentence.", 0.5, 1.0)]
        segment2_words = [MockWord("Second", 1.5, 2.0), MockWord(" sentence.", 2.0, 2.5)]
        
        segment1 = MockSegment("First sentence.", 0.0, 1.0, segment1_words)
        segment2 = MockSegment("Second sentence.", 1.5, 2.5, segment2_words)
        
        mock_model_instance.transcribe.return_value = ([segment1, segment2], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3")
        
        assert "First sentence" in result
        assert "Second sentence" in result

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_srt_format(self, mock_whisper_model):
        """Test transcription with SRT format"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [
            MockWord("Hello", 0.0, 0.5),
            MockWord(" world.", 0.5, 1.0)
        ]
        segment = MockSegment("Hello world.", 0.0, 1.0, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        result = converter.transcribe("test.mp3", format="srt")
        
        # transcribe(format="srt") returns SRT string (VTT converted via webvtt-py)
        assert isinstance(result, str)
        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:01,000" in result
        assert "Hello world" in result

    @patch('src.audio_subtitler.WhisperModel')
    def test_transcribe_vtt_vs_srt_format(self, mock_whisper_model):
        """Test difference between VTT and SRT formats"""
        mock_model_instance = MagicMock()
        mock_whisper_model.return_value = mock_model_instance
        
        words = [MockWord("Test", 0.0, 0.5)]
        segment = MockSegment("Test", 0.0, 0.5, words)
        mock_model_instance.transcribe.return_value = ([segment], None)
        
        converter = AudioSubtitler(model_size_or_path="base")
        
        # Get VTT format (stt2vtt)
        vtt_result = converter.transcribe("test.mp3", format="vtt")
        assert isinstance(vtt_result, str)
        assert "WEBVTT" in vtt_result
        assert "00:00:00.000" in vtt_result  # Period separator
        
        # Get SRT format (VTT -> SRT via webvtt-py)
        srt_result = converter.transcribe("test.mp3", format="srt")
        assert isinstance(srt_result, str)
        assert "1\n" in srt_result  # Index
        assert "00:00:00,000" in srt_result  # Comma separator
        assert "WEBVTT" not in srt_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

