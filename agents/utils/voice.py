from typing import Optional, List, Union
from typing import Callable
import logging
from pathlib import Path

import numpy as np
from .utils import VADStatus

try:
    import onnxruntime as ort
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        """enable_vad and enable_wakeword in SpeechToText component requires onnxruntime to be installed. Please install them with `pip install onnxruntime` or `pip install onnxruntime-gpu` for cpu or gpu based deployment.

        For Jetson devices you can download the pre-built ONNX runtime wheels corresponding to your Jetpack version at https://elinux.org/Jetson_Zoo#ONNX_Runtime"""
    ) from e


def _get_onnx_providers(device: str, model: str) -> List[str]:
    """Check for available providers"""
    available = ort.get_available_providers()
    logger = logging.getLogger(model)

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            logger.warning(
                f"CUDA is not available for {model}. Ensure the correct CUDA/cuDNN versions are installed and install ONNX Runtime with `pip install onnxruntime-gpu`. Switching to CPU runtime."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if device == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            logger.warning(
                f"Tensorrt is not available for {model}. Ensure the correct CUDA/cuDNN versions are installed and install ONNX Runtime with TensorRT support. Switching to CPU runtime."
            )
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    return ["CPUExecutionProvider"]


class VADIterator:
    """Adapted from https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    Check out https://github.com/snakers4/silero-vad
    Citation:
        @misc{Silero VAD,
          author = {Silero Team},
          title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
          year = {2024},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {https://github.com/snakers4/silero-vad},
          commit = {insert_some_commit_here},
          email = {hello@silero.ai}
        }

    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 1000,
        speech_pad_ms: int = 30,
        ncpu: int = 1,
        device: str = "cpu",
    ):
        self.threshold = threshold

        self.sample_rate = np.array(sample_rate).astype(np.int64)

        # Initialize the ONNX model
        sessionOptions = ort.SessionOptions()
        sessionOptions.intra_op_num_threads = ncpu
        sessionOptions.inter_op_num_threads = 1

        providers = _get_onnx_providers(device, "VAD")
        self.model = ort.InferenceSession(
            model_path, sess_options=sessionOptions, providers=providers
        )

        # State variable required by vad model
        self._state = np.zeros((2, 1, 128)).astype("float32")

        self.min_silence_samples = sample_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sample_rate * speech_pad_ms / 1000

        self.reset_states()

    def reset_states(self):
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x_np_32: np.ndarray) -> Optional[VADStatus]:
        """
        x: np.ndarray dtype:int16
            audio chunks
        """
        chunks = np.array_split(x_np_32, 2)
        speech_probs = []
        for chunk in chunks:
            window_size_samples = chunk.shape[0]
            self.current_sample += window_size_samples

            ort_inputs = {
                "input": chunk[None,] / 32768,
                "state": self._state,
                "sr": self.sample_rate,
            }

            out, self._state = self.model.run(None, ort_inputs)
            speech_probs.append(out.squeeze())

        speech_prob = np.mean(speech_probs)

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            return VADStatus.START

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return VADStatus.ONGOING
            else:
                self.temp_end = 0
                self.triggered = False
                return VADStatus.END

        return None


class WakeWordSpotter:
    """
    Open-vocabulary wake word detection using sherpa-onnx's streaming
    KeywordSpotter (default, a small zipformer transducer).

    The wake phrase is plain text, encoded into the bundle's BPE tokens at
    load time, so it can be changed in the config without training a
    per-phrase classifier. Audio is fed block by block (same int16-scale
    float convention as VADIterator) and a detection returns the phrase.
    """

    def __init__(
        self,
        model_path: str,
        phrase: Union[str, List[str]] = "ok robot",
        threshold: float = 0.25,
        score: float = 2.0,
        sample_rate: int = 16000,
        ncpu: int = 1,
        device: str = "cpu",
    ):
        try:
            import sherpa_onnx
            import sentencepiece  # noqa: F401 — used when encoding the phrase
        except ImportError as e:
            raise ImportError(
                "Wake word detection requires sherpa-onnx and sentencepiece. "
                "Install them with: pip install sherpa-onnx sentencepiece"
            ) from e

        model_dir = Path(model_path)

        def find(patterns: List[str]) -> Optional[str]:
            for pattern in patterns:
                for match in sorted(model_dir.glob(pattern)):
                    return str(match)
            return None

        encoder = find(["encoder*.onnx", "*encoder*.onnx"])
        decoder = find(["decoder*.onnx", "*decoder*.onnx"])
        joiner = find(["joiner*.onnx", "*joiner*.onnx"])
        tokens = find(["tokens.txt"])
        if not all([encoder, decoder, joiner, tokens]):
            raise FileNotFoundError(
                f"Could not find a sherpa-onnx keyword spotting bundle in "
                f"'{model_path}'. Expected encoder/decoder/joiner onnx files "
                "and tokens.txt (e.g. sherpa-onnx-kws-zipformer-gigaspeech-3.3M)."
            )

        phrases = [phrase] if isinstance(phrase, str) else list(phrase)
        keywords_file = self._build_keywords_file(model_dir, phrases)

        self._sample_rate = sample_rate
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            keywords_file=keywords_file,
            keywords_threshold=threshold,
            keywords_score=score,
            num_threads=ncpu,
            provider="cuda" if device == "cuda" else "cpu",
        )
        self._stream = self._spotter.create_stream()

    @staticmethod
    def _build_keywords_file(model_dir: Path, phrases: List[str]) -> str:
        """Encode wake phrases into the bundle's BPE tokens and write the
        keywords file sherpa expects (one line per phrase: tokens @phrase)."""
        bpe_model = None
        for match in sorted(model_dir.glob("*.model")):
            bpe_model = str(match)
            break
        if bpe_model is None:
            raise FileNotFoundError(
                f"No BPE model (bpe.model) found in '{model_dir}' — required "
                "to encode custom wake phrases for this bundle."
            )
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "Encoding wake phrases requires sentencepiece. "
                "Install it with: pip install sentencepiece"
            ) from e

        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

        # read the token vocabulary to verify the casing convention
        vocab = set()
        with open(model_dir / "tokens.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    vocab.add(parts[0])

        lines = []
        for phrase in phrases:
            pieces = None
            for candidate in (phrase.upper(), phrase, phrase.lower()):
                encoded = sp.encode_as_pieces(candidate)
                if encoded and all(p in vocab for p in encoded):
                    pieces = encoded
                    break
            if pieces is None:
                raise ValueError(
                    f"The wake phrase '{phrase}' cannot be encoded with this "
                    "bundle's tokens — check that the phrase language matches "
                    "the keyword spotting model."
                )
            # NOTE: sherpa's keywords parser splits on whitespace. The display
            # label after @ must not contain spaces
            label = phrase.replace(" ", "_")
            lines.append(f"{' '.join(pieces)} @{label}")

        keywords_file = model_dir / "agents_keywords.txt"
        keywords_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(keywords_file)

    def process(self, x_np_32: np.ndarray) -> Optional[str]:
        """Feed an audio block and return the detected wake phrase, if any.

        :param x_np_32: int16-scale float32 samples (same convention as
            VADIterator)
        """
        self._stream.accept_waveform(self._sample_rate, x_np_32 / 32768.0)
        detected = None
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result:
                detected = result
                # reset so the same stream can detect again later
                self._spotter.reset_stream(self._stream)
        return detected

    def reset(self):
        """Start a fresh detection stream (e.g. at the end of a speech
        segment) so stale audio does not linger in the decoder state."""
        self._stream = self._spotter.create_stream()


class HypothesisBuffer:
    """A simplified Hypothesis buffer for collection output from a streaming speech to text model based on [whisper_stream](https://github.com/ufal/whisper_streaming). Implements LocalAgreement-n policy as used in CUNI-KIT at IWSLT 2022 etc. before.
        @inproceedings{machacek-etal-2023-turning,
        title = "Turning Whisper into Real-Time Transcription System",
        author = "Mach{\'a}{\v{c}}ek, Dominik  and
          Dabre, Raj  and
          Bojar, Ond{\v{r}}ej",
        editor = "Saha, Sriparna  and
          Sujaini, Herry",
        booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
        month = nov,
        year = "2023",
        address = "Bali, Indonesia",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.ijcnlp-demo.3",
        pages = "17--24",
    }
    """

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0

    def reset(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0

    def insert(self, new):
        # Add new words
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        # Remove up to 5 duplicates if they exist in previously commited
        if self.new and self.commited_in_buffer:
            cn = len(self.commited_in_buffer)
            nn = len(self.new)
            for i in range(1, min(min(cn, nn), 5) + 1):
                c = " ".join(
                    [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
                )
                tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                if c == tail:
                    [repr(self.new.pop(0)) for _ in range(i)]
                    break

    def flush(self):
        commit = []
        # loop to confirm words in transcript received in previous step
        while self.new:
            if not self.buffer:
                break
            na, nb, nt = self.new[0]
            if nt == self.buffer[0][2] and abs(na - self.buffer[0][0]) < 0.2:
                commit.append((na, nb, nt))
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        # commit confirmed words
        self.commited_in_buffer.extend(commit)
        return commit

    def complete(self):
        # send any remaining words in buffer
        return self.buffer
