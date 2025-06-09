import queue
import threading
from io import BytesIO
from typing import Any, Union, Optional, List, Dict, Tuple
import numpy as np
import base64
import time

from ..clients.model_base import ModelClient
from ..config import TextToSpeechConfig
from ..ros import Audio, String, Topic
from ..utils import validate_func_args
from .model_component import ModelComponent
from .component_base import ComponentRunType


class TextToSpeech(ModelComponent):
    """
    This component takes in text input and outputs an audio representation of the text using TTS models (e.g. SpeechT5). The generated audio can be played using any audio playback device available on the agent.

    :param inputs: The input topics for the TTS.
        This should be a list of Topic objects, limited to String type.
    :type inputs: list[Topic]
    :param outputs: Optional output topics for the TTS.
        This should be a list of Topic objects, Audio type is handled automatically.
    :type outputs: list[Topic]
    :param model_client: The model client for the TTS.
        This should be an instance of ModelClient.
    :type model_client: ModelClient
    :param config: The configuration for the TTS.
        This should be an instance of TextToSpeechConfig. If not provided, it defaults to TextToSpeechConfig()
    :type config: Optional[TextToSpeechConfig]
    :param trigger: The trigger value or topic for the TTS.
        This can be a single Topic object or a list of Topic objects.
    :type trigger: Union[Topic, list[Topic]
    :param component_name: The name of the TTS component. This should be a string.
    :type component_name: str

    Example usage:
    ```python
    text_topic = Topic(name="text", msg_type="String")
    audio_topic = Topic(name="audio", msg_type="Audio")
    config = TextToSpeechConfig(play_on_device=True)
    model_client = ModelClient(model=SpeechT5(name="speecht5"))
    tts_component = TextToSpeech(
        inputs=[text_topic],
        outputs=[audio_topic],
        model_client=model_client,
        config=config,
        component_name='tts_component'
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: Optional[List[Topic]] = None,
        model_client: ModelClient,
        config: Optional[TextToSpeechConfig] = None,
        trigger: Union[Topic, List[Topic]],
        component_name: str,
        **kwargs,
    ):
        self.config: TextToSpeechConfig = config or TextToSpeechConfig()
        self.allowed_inputs = {"Required": [String]}
        self.handled_outputs = [Audio]

        if isinstance(trigger, float):
            raise TypeError(
                "TextToSpeech component cannot be started as a timed component"
            )
        self.model_client = model_client

        super().__init__(
            inputs,
            outputs,
            model_client,
            self.config,
            trigger,
            component_name,
            **kwargs,
        )

    def custom_on_configure(self):
        # Configure component
        super().custom_on_configure()

        # If play_on_device is enabled, start a playing stream on a separate thread
        if self.config.play_on_device:
            self.queue = queue.Queue(maxsize=self.config.buffer_size)
            self.event = threading.Event()

    def custom_on_deactivate(self):
        if self.config.play_on_device:
            # If play_on_device is enabled, stop the playing stream thread
            self.event.set()

        # Deactivate component
        super().custom_on_deactivate()

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for TextToSpeech models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """

        # set query as trigger
        trigger = kwargs.get("topic")
        if not trigger:
            self.get_logger().error(
                "Trigger topic not found. TextToSpeech component needs to be given a valid trigger topic."
            )
            return None
        query = self.trig_callbacks[trigger.name].get_output()
        if not query:
            return None

        return {"query": query, **self.inference_params}

    def _stream_callback(
        self, _: bytes, frames: int, time_info: Dict, status: int
    ) -> Tuple[bytes, int]:
        """
        Stream callback for PyAudio, consuming NumPy arrays from the queue.
        """
        try:
            import pyaudio
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "play_on_device device configuration for TextToSpeech component requires soundfile and pyaudio modules to be installed. Please install them with `pip install soundfile pyaudio`"
            ) from e

        assert frames == self.config.block_size
        if status:
            if pyaudio.paOutputUnderflow:
                self.get_logger().warn(
                    "Output underflow: Try to increase the blocksize. Default is 1024"
                )
            else:
                self.get_logger().warn(f"PyAudio stream status flags: {status}")

        # Bytes PyAudio expects = requested_frames * channels * bytes_per_sample
        expected_bytes_len = (
            frames * self._current_channels * 4  # float32 is 4 bytes per sample
        )
        try:
            # get numpy chunk from soundfile
            data = self.queue.get_nowait()
        except queue.Empty:
            self.get_logger().warn(
                "Buffer is empty: If playback was not completed then try to increase the buffersize. Default is 20 (blocks)"
            )
            self.event.set()
            return (b"\x00" * expected_bytes_len, pyaudio.paComplete)

        # If last chunk is smaller than the full block then pad
        if data.shape[0] < frames:
            # create padding array of zeros
            padding_frames = frames - data.shape[0]
            padding_np = np.zeros(
                (padding_frames, self._current_channels), dtype=data.dtype
            )
            # concatenate the actual data with padding
            final_data_np = np.concatenate((data, padding_np), axis=0)
            out_data_bytes = final_data_np.tobytes()
            self.event.set()  # signal that we've processed the true end of data.
            return out_data_bytes, pyaudio.paComplete
        else:
            out_data_bytes = data.tobytes()
            return out_data_bytes, pyaudio.paContinue

    def __feed_data(self, stream, blocks, timeout: int):
        """Feed blocks to playback stream"""
        for data in blocks:
            try:
                self.queue.put(data, timeout=timeout)
            except queue.Full:
                self.get_logger().warn(
                    "Queue full while feeding stream. Playback might be choppy."
                )
                if stream and stream.is_active():
                    time.sleep(timeout / 10)
                else:
                    break
            if self.event.is_set():
                self.get_logger().debug("Event set, stopping data feed.")
                break

        # Wait until playback is finished after last chunck
        wait_start_time = time.monotonic()
        estimated_remaining_blocks = self.queue.qsize() + 10
        max_wait_timeout = estimated_remaining_blocks * (
            self.config.block_size / self._current_framerate
        )
        max_wait_timeout = max(max_wait_timeout, 2.0)
        max_wait_timeout = min(max_wait_timeout, 60.0)  # Cap timeout

        while stream and stream.is_active():
            if self.event.is_set():
                break
            if time.monotonic() - wait_start_time > max_wait_timeout:
                self.get_logger().warn(
                    f"Timeout ({max_wait_timeout:.2f}s) waiting for stream to finish."
                )
                self.event.set()
                break
            time.sleep(0.05)

    def _playback_audio(self, output: Union[bytes, str]):
        """Creates a stream to play audio on device

        :param output:
        :type output: bytes
        """
        # import packages
        try:
            from soundfile import SoundFile
            import pyaudio
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "play_on_device device configuration for TextToSpeech component requires soundfile and pyaudio modules to be installed. Please install them with `pip install soundfile pyaudio`"
            ) from e

        # change str to bytes if output is str
        if isinstance(output, str):
            try:
                output_bytes = base64.b64decode(output)
            except Exception as e:
                output_bytes = b""
                self.get_logger().error(f"Failed to decode base64 string: {e}")
                self.event.set()
        else:
            output_bytes = output

        # clear any set event
        self.event.clear()

        with SoundFile(BytesIO(output_bytes)) as f:
            self._current_channels = f.channels
            self._current_framerate = f.samplerate

            # make chunk generator
            # request float32 from sound, pyAudio paFloat32 corresponds to this
            blocks = f.blocks(self.config.block_size, dtype="float32", always_2d=True)

            # pre-fill queue
            for _ in range(self.config.buffer_size):
                try:
                    data = next(blocks)
                except Exception:
                    break
                if not len(data):
                    break
                self.queue.put_nowait(data)

            # initialize pyaudi stream
            audio_interface = pyaudio.PyAudio()
            try:
                stream = audio_interface.open(
                    format=pyaudio.paFloat32,
                    channels=self._current_channels,
                    rate=self._current_framerate,
                    output=True,
                    frames_per_buffer=self.config.block_size,
                    stream_callback=self._stream_callback,  # type: ignore
                    output_device_index=self.config.device,
                )
                stream.start_stream()
                self.get_logger().debug(
                    "PyAudio stream started. Feeding data using SoundFile blocks..."
                )
                timeout = (
                    self.config.block_size * self.config.buffer_size / f.samplerate
                )

                # Feed data to the device stream
                self.__feed_data(stream, blocks, timeout)

                # Stop stream
                if stream:
                    stream.stop_stream()
                    stream.close()
                    audio_interface.terminate()

            except Exception as e:
                self.get_logger().error(f"PyAudio stream failed: {e}")
                audio_interface.terminate()
                self.event.set()
                return

    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """

        if self.run_type is ComponentRunType.EVENT:
            trigger = kwargs.get("topic")
            if not trigger:
                return
            self.get_logger().debug(f"Received trigger on topic {trigger.name}")
        else:
            time_stamp = self.get_ros_time().sec
            self.get_logger().debug(f"Sending at {time_stamp}")

        # create inference input
        inference_input = self._create_input(*args, **kwargs)
        # call model inference
        if not inference_input:
            self.get_logger().warning("Input not received, not calling model inference")
            return

        # conduct inference
        result = self.model_client.inference(inference_input)
        if result:
            if self.config.play_on_device:
                # Stop any previous playback by setting event and clearing queue
                self.event.set()
                with self.queue.mutex:
                    self.queue.queue.clear()
                # Start a new playback thread
                threading.Thread(
                    target=self._playback_audio,
                    args=(result.get("output"),),
                    daemon=True,
                ).start()

            # publish inference result
            for publisher in self.publishers_dict.values():
                publisher.publish(**result)
        else:
            # raise a fallback trigger via health status
            self.health_status.set_failure()

    def _warmup(self):
        """Warm up and stat check"""
        import time

        inference_input = {
            "query": "Add the sum to the product of these three.",
            **self.inference_params,
        }

        # Run inference once to warm up and once to measure time
        self.model_client.inference(inference_input)

        start_time = time.time()
        self.model_client.inference(inference_input)
        elapsed_time = time.time() - start_time

        self.get_logger().warning(f"Approximate Inference time: {elapsed_time} seconds")
        self.get_logger().warning(
            f"RTF: {elapsed_time / 2}"  # approx audio length, 2 seconds
        )
