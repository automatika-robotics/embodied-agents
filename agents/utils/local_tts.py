"""Local TTS wrapper using sherpa-onnx."""

import io
import queue
import threading
import wave
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np

# Model families supported by sherpa-onnx and the bundle files that identify
# them (first match wins).
_MODEL_FAMILIES: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "supertonic": {
        "required": {
            "tts_json": ["tts.json"],
            "duration_predictor": ["duration_predictor*.onnx"],
            "text_encoder": ["text_encoder*.onnx"],
            "unicode_indexer": ["unicode_indexer*.onnx"],
            "vector_estimator": ["vector_estimator*.onnx"],
            "vocoder": ["vocoder*.onnx"],
        },
        "optional": {"voice_style": ["voice_style*.bin", "voice_style*.json"]},
    },
    "pocket": {
        "required": {
            "lm_main": ["lm_main.onnx", "lm_main.int8.onnx"],
            "lm_flow": ["lm_flow.onnx", "lm_flow.int8.onnx"],
            "encoder": ["encoder.onnx", "encoder.int8.onnx"],
            "decoder": ["decoder.onnx", "decoder.int8.onnx"],
            "text_conditioner": [
                "text_conditioner.onnx",
                "text_conditioner.int8.onnx",
            ],
            "vocab_json": ["vocab.json"],
            "token_scores_json": ["token_scores.json"],
        },
        "optional": {},
    },
    "matcha": {
        "required": {
            "acoustic_model": [
                "matcha*.onnx",
                "model-steps*.onnx",
                "acoustic*.onnx",
            ],
            "vocoder": ["*vocos*.onnx", "*hifigan*.onnx", "vocoder*.onnx"],
            "tokens": ["tokens.txt"],
        },
        "optional": {
            "lexicon": ["lexicon*.txt"],
            "data_dir": ["espeak-ng-data"],
            "dict_dir": ["dict"],
        },
    },
    "zipvoice": {
        "required": {
            "encoder": ["encoder*.onnx"],
            "decoder": ["decoder*.onnx"],
            "vocoder": ["vocoder*.onnx", "*vocos*.onnx"],
            "tokens": ["tokens.txt"],
        },
        "optional": {
            "lexicon": ["lexicon*.txt"],
            "data_dir": ["espeak-ng-data"],
        },
    },
    "kitten": {
        "required": {
            "model": ["model.onnx", "model.int8.onnx"],
            "voices": ["voices.bin"],
            "tokens": ["tokens.txt"],
        },
        "optional": {"data_dir": ["espeak-ng-data"]},
    },
    "kokoro": {
        "required": {
            "model": ["model.onnx", "model.int8.onnx", "kokoro*.onnx"],
            "voices": ["voices.bin"],
            "tokens": ["tokens.txt"],
        },
        "optional": {
            "data_dir": ["espeak-ng-data"],
            "dict_dir": ["dict"],
            "lexicon": ["lexicon*.txt"],
        },
    },
    "vits": {
        "required": {
            "model": ["model.onnx", "model.int8.onnx", "*.onnx"],
            "tokens": ["tokens.txt"],
        },
        "optional": {
            "lexicon": ["lexicon*.txt"],
            "data_dir": ["espeak-ng-data"],
            "dict_dir": ["dict"],
        },
    },
}

# Options that apply to sherpa-onnx's top-level OfflineTtsConfig
_TOP_LEVEL_OPTIONS = ("rule_fsts", "rule_fars", "max_num_sentences", "silence_scale")

# Options that apply to sherpa-onnx's per-call GenerationConfig (used by
# e.g. pocket/zipvoice). 'voice' is a wav path or a name resolved in the bundle
_GENERATION_OPTIONS = (
    "voice",
    "num_steps",
    "reference_text",
    "max_reference_audio_len",
)

# Families that require a reference voice wav for generation
_REFERENCE_VOICE_FAMILIES = ("pocket", "zipvoice")

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
    """Detect the sherpa-onnx TTS model family from a bundle's contents.

    :param model_path: Path to the downloaded model directory.
    :param model_type: Optional family name to force (skips detection).
    :return: Tuple of (family name, resolved file fields for the sub-config).
    :raises ValueError: If the family cannot be determined or forced files
        are missing.
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
        # NOTE: kokoro and kitten bundles share the same file layout. Without an
        # explicit model_type, kitten is only selected on a name hint
        if family == "kitten" and model_type is None and "kitten" not in dir_hint:
            continue
        spec = _MODEL_FAMILIES[family]
        resolved = {}
        for field, patterns in spec["required"].items():
            found = _resolve_file(model_dir, patterns)
            if found is None:
                break
            resolved[field] = found
        else:
            for field, patterns in spec["optional"].items():
                found = _resolve_file(model_dir, patterns)
                if found is not None:
                    resolved[field] = found
            return family, resolved

    raise ValueError(
        f"Could not detect a sherpa-onnx TTS model family from the contents of "
        f"'{model_path}'. Supported families: {sorted(_MODEL_FAMILIES)}. If the "
        f"bundle belongs to one of these families, force it by setting "
        f"local_model_options={{'{_MODEL_TYPE_KEY}': '<family>'}} in the config."
    )


def split_model_options(sub_config_cls, options: Dict) -> Tuple[Dict, Dict, Dict]:
    """Validate user options against the detected family's config fields.
    Unrecognized key raises with the valid keys for the family.

    :param sub_config_cls: The sherpa-onnx OfflineTts*ModelConfig class.
    :param options: User-provided local_model_options (model_type excluded).
    :return: Tuple of (sub-config, top-level, generation options).
    """
    probe = sub_config_cls()
    valid_sub = [
        attr
        for attr in dir(probe)
        if not attr.startswith("_") and not callable(getattr(probe, attr))
    ]
    sub_options, top_options, generation_options = {}, {}, {}
    for key, value in options.items():
        if key in valid_sub:
            sub_options[key] = value
        elif key in _TOP_LEVEL_OPTIONS:
            top_options[key] = value
        elif key in _GENERATION_OPTIONS:
            generation_options[key] = value
        else:
            raise ValueError(
                f"Unknown local_model_options key '{key}' for this model family. "
                f"Valid family options: {sorted(valid_sub)}. "
                f"Valid top-level options: {sorted(_TOP_LEVEL_OPTIONS)}. "
                f"Valid generation options: {sorted(_GENERATION_OPTIONS)}. "
                f"Reserved: '{_MODEL_TYPE_KEY}'."
            )
    return sub_options, top_options, generation_options


def resolve_voice_wav(
    model_dir: Union[str, Path], voice: Optional[str]
) -> Optional[str]:
    """Resolve a reference-voice wav for voice-prompted families.

    :param model_dir: The model bundle directory.
    :param voice: A wav path, or a name resolved inside the bundle (e.g.
        'loona' matches 'test_wavs/loona.wav'). When None, the first wav
        shipped in the bundle is used.
    :return: Path to the wav, or None if nothing was found.
    """
    model_dir = Path(model_dir)
    if voice:
        candidate = Path(voice)
        if candidate.is_file():
            return str(candidate)
        for match in sorted(model_dir.rglob(f"{voice}*.wav")):
            return str(match)
        raise ValueError(
            f"Voice '{voice}' is neither a wav file nor a name found in the "
            f"model bundle at '{model_dir}'. Available bundle voices: "
            f"{[p.stem for p in sorted(model_dir.rglob('*.wav'))]}"
        )
    for pattern in ("test_wavs/*.wav", "*.wav"):
        for match in sorted(model_dir.glob(pattern)):
            return str(match)
    return None


def _load_voice_wav(path: str) -> Tuple[np.ndarray, int]:
    """Load a reference wav as float32 samples in [-1, 1] plus sample rate."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"Reference voice wav '{path}' must be 16-bit PCM "
                f"(got sample width {wf.getsampwidth()} bytes)"
            )
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        channels = wf.getnchannels()
        if channels > 1:
            samples = samples.reshape(-1, channels)[:, 0].copy()
        return samples, wf.getframerate()


class LocalTTS:
    """Local Text-to-Speech inference using sherpa-onnx.

    The model family (pocket, kokoro, kitten, vits, matcha, supertonic,
    zipvoice) is detected automatically from the contents of the model
    directory.

    :param model_path: Path to the model directory containing the bundle files
    :param device: Device to run on ('cpu' or 'cuda')
    :param ncpu: Number of CPU threads
    :param speaker_id: Voice index passed to generation (multi-voice models)
    :param stream: Yield audio chunks as they are synthesized instead of a
        single WAV (uses sherpa-onnx's generation callback). Default True,
        matching the component's stream default
    :param model_options: Options applied to the detected family's sub-config
        or the top-level OfflineTtsConfig, validated against the available
        fields. The reserved key 'model_type' forces the model family.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        ncpu: int = 1,
        speaker_id: int = 0,
        stream: bool = True,
        model_options: Optional[Dict] = None,
    ):
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(
                "Local TTS model deployment requires sherpa-onnx. "
                "Install it with: pip install sherpa-onnx"
            ) from e

        self.device = device
        self.ncpu = ncpu
        self.speaker_id = speaker_id
        self.stream = stream

        options = dict(model_options or {})
        model_type = options.pop(_MODEL_TYPE_KEY, None)
        # detect family
        family, file_fields = detect_model_family(model_path, model_type)

        sub_config_cls = getattr(
            sherpa_onnx, f"OfflineTts{family.capitalize()}ModelConfig"
        )
        sub_options, top_options, generation_options = split_model_options(
            sub_config_cls, options
        )

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                **{family: sub_config_cls(**{**file_fields, **sub_options})},
                num_threads=ncpu,
                provider="cuda" if device == "cuda" else "cpu",
            ),
            **top_options,
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self.model_family = family
        self._generation_config = self._build_generation_config(
            sherpa_onnx, model_path, family, generation_options, top_options
        )

    def _build_generation_config(
        self, sherpa_onnx, model_path, family, generation_options, top_options
    ):
        """Build a per-call GenerationConfig for voice-prompted families.

        Families like pocket and zipvoice synthesize against a reference
        voice wav; other families return None here and use the plain
        (sid, speed) generation path.
        """
        needs_voice = family in _REFERENCE_VOICE_FAMILIES
        if not needs_voice and not generation_options:
            return None

        voice_wav = resolve_voice_wav(model_path, generation_options.pop("voice", None))
        if needs_voice and voice_wav is None:
            raise ValueError(
                f"The '{family}' model family needs a reference voice wav and "
                f"the bundle at '{model_path}' does not ship one. Provide one "
                "with local_model_options={'voice': '/path/to/voice.wav'}."
            )

        generation_config = sherpa_onnx.GenerationConfig()
        generation_config.sid = self.speaker_id
        if voice_wav is not None:
            samples, sample_rate = _load_voice_wav(voice_wav)
            generation_config.reference_audio = samples.tolist()
            generation_config.reference_sample_rate = sample_rate
        if "silence_scale" in top_options:
            generation_config.silence_scale = top_options["silence_scale"]
        if "num_steps" in generation_options:
            generation_config.num_steps = generation_options["num_steps"]
        if "reference_text" in generation_options:
            generation_config.reference_text = generation_options["reference_text"]
        if "max_reference_audio_len" in generation_options:
            generation_config.extra["max_reference_audio_len"] = str(
                generation_options["max_reference_audio_len"]
            )
        return generation_config

    def __call__(self, inference_input: Dict, stream: bool = False) -> Dict:
        """Run TTS inference.

        :param inference_input: Dict with 'query' (text string)
        :param stream: Stream audio chunks (passed by the component from
            config.stream when truthy; the constructor value is the fallback)
        :returns: Dict with 'output' (WAV bytes, or a generator of WAV-chunk
            bytes when streaming is enabled)
        """
        text = inference_input["query"]
        text = text.strip() if text else ""
        if not text:
            return {"output": b""}

        # NOTE: autoregressive models (e.g. pocket) predict when to stop speaking;
        # unpunctuated text destabilizes that prediction. Ensure the
        # text ends with terminal punctuation
        if text[-1] not in ".!?…":
            text += "."

        if stream or self.stream:
            return {"output": self._generate_stream(text)}

        audio = self._generate(text)
        wav_bytes = self._samples_to_wav(audio.samples, audio.sample_rate)

        return {"output": wav_bytes}

    def _generate(self, text: str, callback=None):
        """Generate audio, using the reference-voice path when configured."""
        if self._generation_config is not None:
            if callback is not None:
                return self._tts.generate(text, self._generation_config, callback)
            return self._tts.generate(text, self._generation_config)
        if callback is not None:
            return self._tts.generate(
                text, sid=self.speaker_id, speed=1.0, callback=callback
            )
        return self._tts.generate(text, sid=self.speaker_id, speed=1.0)

    def _generate_stream(self, text: str) -> Generator[bytes, None, None]:
        """Yield WAV-encoded audio chunks as the model synthesizes them.

        sherpa-onnx invokes the generation callback synchronously, so
        generation runs on a worker thread feeding a queue that this
        generator drains.
        """
        chunk_queue: queue.Queue = queue.Queue()
        sample_rate = self._tts.sample_rate

        def _on_chunk(samples, progress) -> int:
            # the callback buffer is reused by sherpa-onnx, copy it
            chunk_queue.put(np.array(samples, dtype=np.float32, copy=True))
            return 1

        def _worker():
            try:
                self._generate(text, callback=_on_chunk)
            finally:
                chunk_queue.put(None)

        # Start thread
        threading.Thread(target=_worker, name="LocalTTS-Generate", daemon=True).start()

        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            if chunk.size:
                yield self._samples_to_wav(chunk, sample_rate)

    @staticmethod
    def _samples_to_wav(samples: Union[np.ndarray, List], sample_rate: int) -> bytes:
        """Convert float32 samples to WAV bytes using stdlib only.

        :param samples: Float32 audio samples in [-1, 1]
        :param sample_rate: Sample rate in Hz
        :returns: WAV file bytes
        """
        samples = np.array(samples, dtype=np.float32)
        int16_data = (samples * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16_data.tobytes())
        return buf.getvalue()
