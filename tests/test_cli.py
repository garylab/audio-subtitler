import pytest
from unittest.mock import patch

from src.cli import main


@patch('src.cli.AudioSubtitler')
@patch('sys.argv', ['audiotovtt', '--help'])
def test_help(mock_audio_subtitler):
    """Test that help message is displayed"""
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
