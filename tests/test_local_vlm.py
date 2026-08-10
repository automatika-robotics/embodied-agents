"""Tests for LocalVLM wrapper — no ROS needed."""

import sys
import threading
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_deps():
    """Mock llama_cpp and its chat format module before importing LocalVLM."""
    mock_llama = MagicMock()
    mock_chat_format = MagicMock()
    mock_llama.llama_chat_format = mock_chat_format
    with patch.dict(
        sys.modules,
        {
            "llama_cpp": mock_llama,
            "llama_cpp.llama_chat_format": mock_chat_format,
        },
    ):
        yield mock_llama, mock_chat_format


@pytest.fixture
def local_vlm(mock_deps):
    mock_llama, mock_chat_format = mock_deps
    from agents.utils.local_vlm import LocalVLM

    vlm = LocalVLM.__new__(LocalVLM)
    vlm.llm = MagicMock()
    vlm.device = "cpu"
    vlm.ncpu = 1
    vlm._lock = threading.Lock()
    return vlm


def _mock_vlm_response(content="A cat"):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class TestLocalVLMCall:
    def test_with_numpy_image(self, local_vlm):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        local_vlm.llm.create_chat_completion.return_value = _mock_vlm_response("A cat")

        result = local_vlm({
            "query": [{"role": "user", "content": "What is this?"}],
            "images": [img],
        })
        assert result["output"] == "A cat"
        local_vlm.llm.create_chat_completion.assert_called_once()

        # Verify the message has multimodal format
        call_kwargs = local_vlm.llm.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert any(item["type"] == "image_url" for item in content)
        assert any(item["type"] == "text" for item in content)

    def test_streaming_returns_generator(self, local_vlm):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "A red"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " box"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        local_vlm.llm.create_chat_completion.return_value = iter(chunks)

        result = local_vlm(
            {"query": [{"role": "user", "content": "What is this?"}], "images": [img]},
            stream=True,
        )
        assert list(result["output"]) == ["A red", " box"]
        assert local_vlm.llm.create_chat_completion.call_args[1]["stream"] is True

    def test_generation_params_forwarded(self, local_vlm):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        local_vlm.llm.create_chat_completion.return_value = _mock_vlm_response("ok")

        local_vlm({
            "query": [{"role": "user", "content": "What is this?"}],
            "images": [img],
            "temperature": 0.5,
            "max_new_tokens": 100,
        })
        call_kwargs = local_vlm.llm.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    def test_no_images(self, local_vlm):
        result = local_vlm({
            "query": [{"role": "user", "content": "What is this?"}],
            "images": [],
        })
        assert result["output"] == "No image provided."

    def test_extracts_last_user_query(self, local_vlm):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        local_vlm.llm.create_chat_completion.return_value = _mock_vlm_response("ok")

        local_vlm({
            "query": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Second question"},
            ],
            "images": [img],
        })

        call_kwargs = local_vlm.llm.create_chat_completion.call_args[1]
        messages = call_kwargs["messages"]
        # The full conversation is preserved and the images are attached to
        # the LAST user message
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "First question"}
        assert messages[1] == {"role": "assistant", "content": "Answer"}
        text_items = [
            item for item in messages[2]["content"] if item["type"] == "text"
        ]
        assert text_items[0]["text"] == "Second question"

    def test_image_data_uri_format(self, local_vlm):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        local_vlm.llm.create_chat_completion.return_value = _mock_vlm_response("ok")

        local_vlm({
            "query": [{"role": "user", "content": "test"}],
            "images": [img],
        })

        call_kwargs = local_vlm.llm.create_chat_completion.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        image_item = next(item for item in content if item["type"] == "image_url")
        url = image_item["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")


class TestVLMFamilyDetection:
    def test_families_detected_from_name(self):
        from agents.utils.local_vlm import detect_vlm_family

        assert detect_vlm_family("ggml-org/moondream2-20250414-GGUF") == "moondream"
        assert detect_vlm_family("unsloth/Qwen3-VL-2B-Instruct-GGUF") == "qwen_vl"
        assert detect_vlm_family("openbmb/MiniCPM-V-2_6-gguf") == "minicpm"
        assert detect_vlm_family("cjpais/llava-v1.6-mistral-7b-gguf") == "llava16"
        assert detect_vlm_family("mys/ggml_llava-v1.5-7b") == "llava"

    def test_model_type_override_and_unknown(self):
        from agents.utils.local_vlm import detect_vlm_family

        assert detect_vlm_family("some/opaque-model", model_type="llava") == "llava"
        with pytest.raises(ValueError, match="model_type"):
            detect_vlm_family("some/opaque-model")
        with pytest.raises(ValueError, match="Unknown model_type"):
            detect_vlm_family("x", model_type="not_a_family")


class TestVLMMessageBuilding:
    def test_context_preserved_and_multi_image(self, local_vlm):
        local_vlm.llm.create_chat_completion.return_value = _mock_vlm_response(
            "Two oranges"
        )
        images = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
        ]
        result = local_vlm(
            {
                "query": [
                    {"role": "system", "content": "You are a robot."},
                    {"role": "user", "content": "What do you see?"},
                ],
                "images": images,
            }
        )
        assert result["output"] == "Two oranges"
        messages = local_vlm.llm.create_chat_completion.call_args[1]["messages"]
        # system prompt preserved
        assert messages[0] == {"role": "system", "content": "You are a robot."}
        # both images attached to the user message, text kept
        user_content = messages[1]["content"]
        assert sum(1 for part in user_content if part["type"] == "image_url") == 2
        assert user_content[-1] == {"type": "text", "text": "What do you see?"}
