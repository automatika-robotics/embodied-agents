import queue
import threading
from io import BytesIO
from typing import Any, Union, Optional, List, Dict
import numpy as np
import base64

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

        return {"query": query, **self.config._get_inference_params()}

    def _stream_callback(self, outdata: np.ndarray, frames: int, _, status) -> None:
        """Stream callback function for playing audio on device

        :param outdata:
        :type outdata: np.ndarray
        :param frames:
        :type frames: int
        :param _:
        :param status:
        :type status: sd.CallbackFlags
        :rtype: None
        """
        try:
            from sounddevice import CallbackStop
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "play_on_device device configuration for TextToSpeech component requires soundfile and sounddevice modules to be installed. Please install them with `pip install soundfile sounddevice`"
            ) from e
        assert frames == self.config.block_size
        if status.output_underflow:
            self.get_logger().warn(
                "Output underflow: Try to increase the blocksize. Default is 1024"
            )
        try:
            data = self.queue.get_nowait()
        except queue.Empty as e:
            self.get_logger().warn(
                "Buffer is empty: If playback was not completed then try to increase the buffersize. Default is 20 (blocks)"
            )
            raise CallbackStop from e
        if len(data) < len(outdata):
            outdata[: len(data)] = data
            outdata[len(data) :].fill(0)
            raise CallbackStop
        else:
            outdata[:] = data

    def _playback_audio(self, output: Union[bytes, str]) -> None:
        """Creates a stream to play audio on device

        :param output:
        :type output: bytes
        :rtype: None
        """
        # import packages
        try:
            from soundfile import SoundFile
            from sounddevice import OutputStream
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "play_on_device device configuration for TextToSpeech component requires soundfile and sounddevice modules to be installed. Please install them with `pip install soundfile sounddevice`"
            ) from e

        # change str to bytes if output is str
        if isinstance(output, str):
            output = base64.b64decode(output)

        # clear any set event
        self.event.clear()

        with SoundFile(BytesIO(output)) as f:
            # make chunk generator
            blocks = f.blocks(self.config.block_size, always_2d=True)

            # pre-fill queue
            for _ in range(self.config.buffer_size):
                try:
                    data = next(blocks)
                except Exception:
                    break
                if not len(data):
                    break
                self.queue.put_nowait(data)

            # create an output stream
            stream = OutputStream(
                samplerate=f.samplerate,
                blocksize=self.config.block_size,
                device=self.config.device,
                channels=f.channels,
                callback=self._stream_callback,
                finished_callback=self.event.set,
            )

            # invoke stream callback
            with stream:
                timeout = (
                    self.config.block_size * self.config.buffer_size / f.samplerate
                )
                for data in blocks:
                    self.queue.put(data, timeout=timeout)
                    # Stop playback if event is set
                    if self.event.is_set():
                        break
                # Wait until playback is finished after last chunck
                self.event.wait()

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
            **self.config._get_inference_params(),
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
