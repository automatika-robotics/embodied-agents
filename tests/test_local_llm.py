"""Tests for LocalLLM wrapper — no ROS needed."""

import sys
import json
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_llama_cpp():
    """Mock llama_cpp before importing LocalLLM."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock_module}):
        yield mock_module


@pytest.fixture
def local_llm(mock_llama_cpp):
    from agents.utils.local_llm import LocalLLM

    llm = LocalLLM.__new__(LocalLLM)
    llm.llm = MagicMock()
    llm.device = "cpu"
    llm.ncpu = 1
    return llm


def _mock_response(content="Hello there!", tool_calls=None):
    """Build a mock non-streaming chat completion response."""
    message = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"
    return {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _mock_stream_chunks(tokens):
    """Build mock streaming chunks from a list of token strings."""
    chunks = []
    # First chunk: role announcement
    chunks.append({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]})
    # Content chunks
    for token in tokens:
        chunks.append({"choices": [{"delta": {"content": token}, "finish_reason": None}]})
    # Final chunk
    chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    return chunks


class TestCallNonStreaming:
    def test_returns_output(self, local_llm):
        local_llm.llm.create_chat_completion.return_value = _mock_response(
            "Hello there!"
        )

        result = local_llm({"query": [{"role": "user", "content": "Hi"}]})
        assert result["output"] == "Hello there!"
        local_llm.llm.create_chat_completion.assert_called_once()

    def test_passes_temperature_and_max_tokens(self, local_llm):
        local_llm.llm.create_chat_completion.return_value = _mock_response("ok")

        local_llm({
            "query": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_new_tokens": 100,
        })

        call_kwargs = local_llm.llm.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100


class TestCallStreaming:
    def test_returns_generator(self, local_llm):
        chunks = _mock_stream_chunks(["Hello", " world"])
        local_llm.llm.create_chat_completion.return_value = iter(chunks)

        result = local_llm(
            {"query": [{"role": "user", "content": "Hi"}]}, stream=True
        )
        assert "output" in result
        tokens = list(result["output"])
        assert tokens == ["Hello", " world"]


class TestCallWithTools:
    def test_tools_parsed_from_response(self, local_llm):
        tool_calls = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "route_to_nav",
                    "arguments": json.dumps({"x": 1}),
                },
            }
        ]
        local_llm.llm.create_chat_completion.return_value = _mock_response(
            content=None, tool_calls=tool_calls
        )

        tools = [{"type": "function", "function": {"name": "route_to_nav"}}]
        result = local_llm(
            {"query": [{"role": "user", "content": "Go"}], "tools": tools}
        )
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "route_to_nav"
        assert result["tool_calls"][0]["function"]["arguments"] == {"x": 1}

    def test_passes_tools_to_api(self, local_llm):
        local_llm.llm.create_chat_completion.return_value = _mock_response("ok")

        tools = [{"type": "function", "function": {"name": "test_fn"}}]
        local_llm(
            {"query": [{"role": "user", "content": "Hi"}], "tools": tools}
        )

        call_kwargs = local_llm.llm.create_chat_completion.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"


class TestLocalLLMModelOptions:
    @staticmethod
    def _fake_llama(mock_llama_cpp):
        """Install a fake Llama class with a realistic signature, recording
        constructor and from_pretrained calls."""
        calls = {}

        class FakeLlama:
            def __init__(
                self,
                model_path=None,
                n_ctx=512,
                n_gpu_layers=0,
                n_threads=1,
                n_batch=512,
                flash_attn=False,
                chat_format=None,
                verbose=True,
            ):
                calls["init"] = {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": n_gpu_layers,
                    "n_threads": n_threads,
                    "n_batch": n_batch,
                    "flash_attn": flash_attn,
                    "chat_format": chat_format,
                    "verbose": verbose,
                }

            @classmethod
            def from_pretrained(cls, repo_id=None, filename=None, **kwargs):
                calls["from_pretrained"] = {
                    "repo_id": repo_id,
                    "filename": filename,
                    **kwargs,
                }
                return MagicMock()

        mock_llama_cpp.Llama = FakeLlama
        return calls

    def test_options_passed_and_defaults_set(self, mock_llama_cpp, tmp_path):
        calls = self._fake_llama(mock_llama_cpp)
        from agents.utils.local_llm import LocalLLM

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        LocalLLM(
            str(gguf), device="cpu", ncpu=2, model_options={"n_batch": 1024}
        )
        assert calls["init"]["n_batch"] == 1024
        # context defaults to the model's trained length, not llama-cpp's 512
        assert calls["init"]["n_ctx"] == 0
        assert calls["init"]["n_threads"] == 2

    def test_filename_selects_quant_from_repo(self, mock_llama_cpp):
        calls = self._fake_llama(mock_llama_cpp)
        from agents.utils.local_llm import LocalLLM

        LocalLLM(
            "some/repo-GGUF",
            device="cpu",
            model_options={"filename": "*q4_k_m*.gguf"},
        )
        assert calls["from_pretrained"]["filename"] == "*q4_k_m*.gguf"

    def test_unknown_option_raises(self, mock_llama_cpp, tmp_path):
        self._fake_llama(mock_llama_cpp)
        from agents.utils.local_llm import LocalLLM

        gguf = tmp_path / "model.gguf"
        gguf.touch()
        with pytest.raises(ValueError, match="Valid options"):
            LocalLLM(str(gguf), model_options={"not_an_option": 1})
