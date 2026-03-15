import json
import re
from typing import Dict, Generator, Union


class LocalLLM:
    """Local LLM inference using onnxruntime-genai.

    :param model_path: Path to the ONNX model directory
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    """

    def __init__(self, model_path: str, device: str = "cuda", ncpu: int = 1):
        try:
            import onnxruntime_genai as og
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Local LLM model deployment requires onnxruntime-genai. "
                "Install it with: pip install onnxruntime-genai"
            ) from e

        self._og = og
        self.device = device
        self.ncpu = ncpu

        # Use Config to explicitly set the execution provider so that
        # onnxruntime-genai-cuda doesn't try to load CUDA libs when CPU
        # is requested.
        config = og.Config(model_path)
        config.clear_providers()
        if device == "cuda":
            config.append_provider("CUDAExecutionProvider")
        config.append_provider("CPUExecutionProvider")
        self.model = og.Model(config)
        self.tokenizer = og.Tokenizer(self.model)

    # TODO: parameterize template and tool call format
    def _apply_chat_template(self, messages: list) -> str:
        """Apply a standard chat template to messages."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _parse_tool_calls(self, text: str) -> list:
        """Parse standard tool call format from model output."""
        tool_calls = []
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        for match in pattern.finditer(text):
            try:
                call = json.loads(match.group(1).strip())
                tool_calls.append({
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments", {}),
                    }
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return tool_calls

    def _get_params(self, inference_input: Dict, tokens) -> Dict:
        params = self._og.GeneratorParams(self.model)
        params.input_ids = tokens
        if temperature := inference_input.get("temperature"):
            params.set_search_options(temperature=temperature)
        if max_new_tokens := inference_input.get("max_new_tokens"):
            params.set_search_options(max_length=max_new_tokens)
        return params

    def __call__(
        self, inference_input: Dict, stream=False
    ) -> Union[Dict, Generator[str, None, None]]:
        """Run inference and return complete response.

        :param inference_input: Dict with 'query' (messages list) and optional
            'temperature', 'max_new_tokens', 'tools'
        :returns: Dict with 'output' (str) and optionally 'tool_calls'
        """
        prompt = self._apply_chat_template(inference_input["query"])

        # If tools are provided, inject them into the prompt
        if tools := inference_input.get("tools") and not stream:
            tools_text = json.dumps(tools)
            suffix = "<|im_start|>assistant\n"
            if prompt.endswith(suffix):
                prompt = prompt[: -len(suffix)]
            prompt = (
                f"{prompt}"
                f"<|im_start|>system\nAvailable tools:\n{tools_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # Get params
        params = self._get_params(inference_input, self.tokenizer.encode(prompt))

        if stream:
            return {"output": self._stream(params)}

        output_tokens = self.model.generate(params)[0]
        output_text = self.tokenizer.decode(output_tokens)

        # Strip the prompt echo if present
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt) :]
        # Strip end token
        output_text = output_text.replace("<|im_end|>", "").strip()

        result = {"output": output_text}

        # Parse tool calls if tools were provided
        if inference_input.get("tools"):
            tool_calls = self._parse_tool_calls(output_text)
            if tool_calls:
                result["tool_calls"] = tool_calls

        return result

    def _stream(self, params: Dict) -> Generator[str, None, None]:
        """Stream inference output token by token.

        :param inference_input: Dict with 'query' (messages list) and optional params
        :yields: String tokens
        """
        generator = self._og.Generator(self.model, params)
        tokenizer_stream = self.tokenizer.create_stream()

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            text = tokenizer_stream.decode(token)
            # Skip end tokens
            if "<|im_end|>" in text:
                break
            yield text
