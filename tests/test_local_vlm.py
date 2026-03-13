"""Tests for LocalVLM wrapper — no ROS needed."""

import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_deps():
    """Mock moondream and PIL before importing LocalVLM."""
    mock_md = MagicMock()
    mock_pil = MagicMock()
    mock_pil_image = MagicMock()
    mock_pil.Image = mock_pil_image
    with patch.dict(
        sys.modules, {"moondream": mock_md, "PIL": mock_pil, "PIL.Image": mock_pil_image}
    ):
        yield mock_md, mock_pil_image


@pytest.fixture
def local_vlm(mock_deps):
    mock_md, mock_pil_image = mock_deps
    from agents.utils.local_vlm import LocalVLM

    vlm = LocalVLM.__new__(LocalVLM)
    vlm.model = MagicMock()
    vlm.pil_image = mock_pil_image
    vlm.device = "cpu"
    vlm.ncpu = 1
    return vlm


class TestLocalVLMCall:
    def test_with_numpy_image(self, local_vlm):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        local_vlm.model.encode_image.return_value = "encoded"
        local_vlm.model.query.return_value = {"answer": "A cat"}

        result = local_vlm({
            "query": [{"role": "user", "content": "What is this?"}],
            "images": [img],
        })
        assert result["output"] == "A cat"
        local_vlm.pil_image.fromarray.assert_called_once()
        local_vlm.model.encode_image.assert_called_once()

    def test_no_images(self, local_vlm):
        result = local_vlm({
            "query": [{"role": "user", "content": "What is this?"}],
            "images": [],
        })
        assert result["output"] == "No image provided."

    def test_extracts_last_user_query(self, local_vlm):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        local_vlm.model.encode_image.return_value = "enc"
        local_vlm.model.query.return_value = {"answer": "ok"}

        local_vlm({
            "query": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Second question"},
            ],
            "images": [img],
        })
        # The query passed to model.query should be the last user message
        call_args = local_vlm.model.query.call_args
        assert call_args[0][1] == "Second question"
