"""Local STT wrapper using Moonshine via moonshine-onnx."""

from typing import Dict

import numpy as np


class LocalSTT:
    """Local Speech-to-Text inference using Moonshine ONNX.

    :param model_path: Not used directly (moonshine-onnx manages its own models)
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    """

    def __init__(self, model_path: str, device: str = "cuda", ncpu: int = 1):
        try:
            import moonshine_onnx
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Local STT model requires moonshine-onnx. "
                "Install it with: pip install moonshine-onnx"
            ) from e

        self._moonshine = moonshine_onnx
        self.device = device
        self.ncpu = ncpu

    def __call__(self, inference_input: Dict) -> Dict:
        """Run STT inference.

        :param inference_input: Dict with 'query' (audio bytes)
        :returns: Dict with 'output' (transcribed text)
        """
        audio_data = inference_input["query"]

        # Convert bytes to float32 numpy array
        if isinstance(audio_data, (bytes, bytearray)):
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            audio_np = audio_data.astype(np.float32)
        else:
            return {"output": ""}

        # Moonshine expects (samples,) shaped array at 16kHz
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        text = self._moonshine.transcribe(audio_np)

        # moonshine_onnx.transcribe returns a list of strings
        if isinstance(text, list):
            text = " ".join(text)

        return {"output": text.strip()}
