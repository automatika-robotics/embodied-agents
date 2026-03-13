"""Local TTS wrapper using Kokoro via kokoro-onnx."""

import io
from typing import Dict


class LocalTTS:
    """Local Text-to-Speech inference using Kokoro ONNX.

    :param model_path: Not used directly (kokoro-onnx manages its own models)
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    """

    def __init__(self, model_path: str, device: str = "cuda", ncpu: int = 1):
        try:
            from kokoro_onnx import Kokoro
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Local TTS model requires kokoro-onnx. "
                "Install it with: pip install kokoro-onnx"
            ) from e

        self._kokoro = Kokoro(model_path)
        self.device = device
        self.ncpu = ncpu

    def __call__(self, inference_input: Dict) -> Dict:
        """Run TTS inference.

        :param inference_input: Dict with 'query' (text string)
        :returns: Dict with 'output' (WAV bytes)
        """
        try:
            import soundfile as sf
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "LocalTTS requires soundfile for WAV encoding. "
                "Install it with: pip install soundfile"
            ) from e

        text = inference_input["query"]
        if not text:
            return {"output": b""}

        # Generate audio samples and sample rate
        samples, sample_rate = self._kokoro.create(text, voice="af_heart", speed=1.0)

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, samples, sample_rate, format="WAV")
        wav_bytes = wav_buffer.getvalue()

        return {"output": wav_bytes}
