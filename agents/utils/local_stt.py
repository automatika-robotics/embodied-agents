"""Local STT wrapper using sherpa-onnx."""

import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Model families supported by sherpa-onnx and the bundle files that identify
# them. (first match wins)
_MODEL_FAMILIES: Dict[str, Dict] = {
    "moonshine": {
        "factory": "from_moonshine",
        "required": {
            "preprocessor": ["preprocess*.onnx"],
            "encoder": ["encode*.onnx"],
            "uncached_decoder": ["uncached_decode*.onnx"],
            "cached_decoder": ["cached_decode*.onnx"],
            "tokens": ["tokens.txt"],
        },
    },
    "moonshine_v2": {
        "factory": "from_moonshine_v2",
        "required": {
            "encoder": ["encoder_model*.ort", "encoder_model*.onnx"],
            "decoder": ["decoder_model*.ort", "decoder_model*.onnx"],
            "tokens": ["tokens.txt"],
        },
    },
    "qwen3_asr": {
        "factory": "from_qwen3_asr",
        "required": {
            "conv_frontend": ["conv_frontend*.onnx"],
            "encoder": ["encoder*.onnx"],
            "decoder": ["decoder*.onnx"],
            "tokenizer": ["tokenizer"],
        },
    },
    "transducer": {
        "factory": "from_transducer",
        "required": {
            "encoder": ["*encoder*.onnx"],
            "decoder": ["*decoder*.onnx"],
            "joiner": ["*joiner*.onnx"],
            "tokens": ["tokens.txt"],
        },
    },
    "sense_voice": {
        "factory": "from_sense_voice",
        "name_hints": ("sense-voice", "sense_voice"),
        "required": {"model": ["model*.onnx"], "tokens": ["tokens.txt"]},
    },
    "paraformer": {
        "factory": "from_paraformer",
        "name_hints": ("paraformer",),
        "required": {
            "paraformer": ["model*.onnx", "*paraformer*.onnx"],
            "tokens": ["tokens.txt"],
        },
    },
    "omnilingual_asr_ctc": {
        "factory": "from_omnilingual_asr_ctc",
        "name_hints": ("omnilingual",),
        "required": {"model": ["model*.onnx"], "tokens": ["tokens.txt"]},
    },
    "nemo_ctc": {
        "factory": "from_nemo_ctc",
        "name_hints": ("ctc",),
        "required": {"model": ["model*.onnx"], "tokens": ["tokens.txt"]},
    },
    # Fallback family matched at the end, so any bundle with a plain
    # encoder/decoder/tokens layout keeps loading as whisper
    "whisper": {
        "factory": "from_whisper",
        "required": {
            "encoder": ["*encoder*.onnx"],
            "decoder": ["*decoder*.onnx"],
            "tokens": ["*tokens*.txt"],
        },
    },
}

# NeMo-style transducers need model_type='nemo_transducer'
_NEMO_TRANSDUCER_HINTS = ("nemo", "parakeet")

# Reserved key in local_model_options that forces the model family instead of
# detecting it from the bundle contents
_MODEL_TYPE_KEY = "model_type"


def _resolve_file(model_dir: Path, patterns: List[str]) -> Optional[str]:
    """Return the first file (or directory) in model_dir matching patterns."""
    for pattern in patterns:
        for match in sorted(model_dir.glob(pattern)):
            return str(match)
    return None


def detect_model_family(
    model_path: str, model_type: Optional[str] = None
) -> Tuple[str, Dict[str, str]]:
    """Detect the sherpa-onnx STT model family from a bundle's contents.

    :param model_path: Path to the downloaded model directory.
    :param model_type: Optional family name to force (skips detection).
    :return: Tuple of (family name, resolved file fields for the factory).
    :raises ValueError: If the family cannot be determined.
    """
    model_dir = Path(model_path)

    candidates = list(_MODEL_FAMILIES.keys())
    if model_type is not None:
        if model_type not in _MODEL_FAMILIES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Available families: "
                f"{sorted(_MODEL_FAMILIES)}"
            )
        candidates = [model_type]

    dir_hint = model_dir.name.lower()
    for family in candidates:
        spec = _MODEL_FAMILIES[family]
        hints = spec.get("name_hints")
        if hints and model_type is None and not any(h in dir_hint for h in hints):
            continue
        resolved = {}
        for field, patterns in spec["required"].items():
            found = _resolve_file(model_dir, patterns)
            if found is None:
                break
            resolved[field] = found
        else:
            return family, resolved

    raise ValueError(
        f"Could not detect a sherpa-onnx STT model family from the contents of "
        f"'{model_path}'. Supported families: {sorted(_MODEL_FAMILIES)}. If the "
        f"bundle belongs to one of these families, force it by setting "
        f"local_model_options={{'{_MODEL_TYPE_KEY}': '<family>'}} in the config."
    )


class LocalSTT:
    """Local Speech-to-Text inference using sherpa-onnx.

    The model family (transducer incl. NeMo/parakeet, whisper, qwen3_asr,
    moonshine, sense_voice, paraformer, nemo_ctc, omnilingual) is detected
    automatically from the contents of the model directory, so any
    sherpa-onnx compatible offline STT bundle can be loaded by pointing
    ``local_model_path`` at its HuggingFace repository or a local directory.

    :param model_path: Path to the model directory containing the bundle files
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    :param sample_rate: Sample rate of the incoming audio
    :param language: Language code passed to factories that accept one
        (whisper, sense_voice); ignored by language-agnostic families
    :param tail_paddings: Tail padding frames for whisper models
    :param model_options: Additional keyword options for the detected
        family's sherpa-onnx factory, validated against its signature (e.g.
        ``decoding_method``, ``hotwords_file`` for transducers, ``task`` for
        whisper). The reserved key 'model_type' forces the model family.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        ncpu: int = 1,
        sample_rate: int = 16000,
        language: str = "en",
        tail_paddings: int = 1500,
        model_options: Optional[Dict] = None,
    ):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(
                "Local STT model deployment requires sherpa-onnx. "
                "Install it with: pip install sherpa-onnx"
            ) from e

        self.device = device
        self.ncpu = ncpu
        self._sample_rate = sample_rate

        options = dict(model_options or {})
        model_type = options.pop(_MODEL_TYPE_KEY, None)
        # transducer family (normally auto-detected from the model path)
        transducer_variant = None
        if model_type == "nemo_transducer":
            transducer_variant = model_type
            model_type = "transducer"
        family, file_fields = detect_model_family(model_path, model_type)
        self.model_family = family

        factory = getattr(
            sherpa_onnx.OfflineRecognizer, _MODEL_FAMILIES[family]["factory"]
        )
        factory_params = inspect.signature(factory).parameters

        kwargs = dict(file_fields)
        if "num_threads" in factory_params:
            kwargs["num_threads"] = ncpu
        if "provider" in factory_params:
            kwargs["provider"] = "cuda" if device == "cuda" else "cpu"
        if "language" in factory_params and language:
            kwargs["language"] = language
        if "tail_paddings" in factory_params:
            kwargs["tail_paddings"] = tail_paddings
        if family == "transducer":
            # NOTE: Check the whole path. HF cache paths carry the repo name in a
            # parent component while the leaf is a commit hash
            if transducer_variant is None and any(
                hint in str(Path(model_path).resolve()).lower()
                for hint in _NEMO_TRANSDUCER_HINTS
            ):
                transducer_variant = "nemo_transducer"
            if transducer_variant:
                kwargs["model_type"] = transducer_variant

        # Validate user options against the factory signature
        for key, value in options.items():
            if key not in factory_params or key in file_fields:
                valid = sorted(
                    p for p in factory_params if p not in file_fields and p != "cls"
                )
                raise ValueError(
                    f"Unknown local_model_options key '{key}' for the "
                    f"'{family}' model family. Valid options: {valid}. "
                    f"Reserved: '{_MODEL_TYPE_KEY}'."
                )
            kwargs[key] = value

        self._recognizer = factory(**kwargs)

    def __call__(self, inference_input: Dict) -> Dict:
        """Run STT inference.

        :param inference_input: Dict with 'query' (audio bytes or numpy array)
        :returns: Dict with 'output' (transcribed text)
        """
        audio_data = inference_input["query"]

        # Convert bytes to float32 numpy array
        if isinstance(audio_data, (bytes, bytearray)):
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )
        elif isinstance(audio_data, np.ndarray):
            audio_np = audio_data.astype(np.float32)
        else:
            return {"output": ""}

        # Ensure 1D array at 16kHz
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()

        stream = self._recognizer.create_stream()
        stream.accept_waveform(self._sample_rate, audio_np)
        self._recognizer.decode_stream(stream)

        return {"output": stream.result.text.strip()}
