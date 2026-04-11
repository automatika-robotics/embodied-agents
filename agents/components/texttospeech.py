import queue
import socket
import threading
from io import BytesIO
from typing import Any, Union, Optional, List, Dict
import base64
import time

from ..clients.model_base import ModelClient
from ..clients import RoboMLWSClient, RoboMLRESPClient
from ..config import TextToSpeechConfig
from ..ros import Audio, String, Topic, StreamingString, component_action
from ..utils import validate_func_args, load_model_repo
from .model_component import ModelComponent
from .component_base import ComponentRunType


class TextToSpeech(ModelComponent):
    """
    This component takes in text input and outputs an audio representation of the text using TTS models (e.g. TransformersTTS). The generated audio can be played using any audio playback device available on the agent.

    :param inputs: The input topics for the TTS.
        This should be a list of Topic objects, limited to String type.
    :type inputs: list[Topic]
    :param outputs: Optional output topics for the TTS.
        This should be a list of Topic objects, Audio type is handled automatically.
    :type outputs: list[Topic]
    :param model_client: The model client for the TTS.
        This should be an instance of ModelClient. Optional if ``enable_local_model`` is set to True in the config.
    :type model_client: Optional[ModelClient]
    :param config: The configuration for the TTS.
        This should be an instance of TextToSpeechConfig. If not provided, it defaults to TextToSpeechConfig()
    :type config: Optional[TextToSpeechConfig]
    :param trigger: The trigger value or topic for the TTS.
        This can be a single Topic object or a list of Topic objects.
    :type trigger: Union[Topic, list[Topic]]
    :param component_name: The name of the TTS component. This should be a string.
    :type component_name: str

    Example usage:
    ```python
    text_topic = Topic(name="text", msg_type="String")
    audio_topic = Topic(name="audio", msg_type="Audio")
    config = TextToSpeechConfig(play_on_device=True)
    model_client = ModelClient(model=TransformersTTS(name="tts"))
    tts_component = TextToSpeech(
        inputs=[text_topic],
        outputs=[audio_topic],
        model_client=model_client,
        config=config,
        component_name='tts_component'
    )
    ```

    Example usage with local model:
    ```python
    text_topic = Topic(name="text", msg_type="String")
    config = TextToSpeechConfig(enable_local_model=True, play_on_device=True)
    tts_component = TextToSpeech(
        inputs=[text_topic],
        config=config,
        trigger=text_topic,
        component_name='local_tts'
    )
    ```
    """

    @validate_func_args
    def __init__(
        self,
        *,
        inputs: List[Topic],
        outputs: Optional[List[Topic]] = None,
        model_client: Optional[ModelClient] = None,
        config: Optional[TextToSpeechConfig] = None,
        trigger: Union[Topic, List[Topic]],
        component_name: str,
        **kwargs,
    ):
        self.config: TextToSpeechConfig = config or TextToSpeechConfig()
        self.allowed_inputs = {"Required": [[String, StreamingString]]}
        self.handled_outputs = [Audio]

        if isinstance(trigger, float):
            raise TypeError(
                "TextToSpeech component cannot be started as a timed component"
            )

        if not model_client and not self.config.enable_local_model:
            raise TypeError(
                "TextToSpeech component requires a model_client or enable_local_model=True in TextToSpeechConfig."
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
        # deploy local TTS if enabled
        if not self.model_client and self.config.enable_local_model:
            self._deploy_local_model()
        # Configure component
        super().custom_on_configure()

        # If play_on_device is enabled, start a playing stream on a separate thread
        if self.config.play_on_device:
            self._incoming_queue = queue.Queue()
            self._stop_event = threading.Event()
            self._playback_thread: Optional[threading.Thread] = None
            self._thread_lock = threading.Lock()

            if self.config.stream_to_ip and self.config.stream_to_port:
                self.get_logger().info(
                    f"""UDP streaming configured for target {self.config.stream_to_ip}:{self.config.stream_to_port}. Stream can be played at the target machine with gstreamer or ffmpeg, audio format example with gstreamer is shown below:
        gst-launch-1.0 -v udpsrc port={self.config.stream_to_port} caps="audio/x-raw,format=F32LE,channels=1,rate=16000" ! queue ! audioconvert ! audioresample ! alsasink
        """
                )

        # Get bytes as output from server if using appropriate client
        if isinstance(self.model_client, (RoboMLWSClient, RoboMLRESPClient)):
            self.inference_params["get_bytes"] = True

    def custom_on_deactivate(self):
        if self.config.play_on_device:
            # If play_on_device is enabled, stop the playing stream thread
            self._stop_event.set()

        # Deactivate component
        super().custom_on_deactivate()

    def _deploy_local_model(self):
        """Deploy local TTS model on demand."""
        if self.local_model is not None:
            return  # already deployed
        from ..utils.local_tts import LocalTTS

        self.local_model = LocalTTS(
            model_path=load_model_repo("local_tts", self.config.local_model_path),
            device=self.config.device_local_model,
            ncpu=self.config.ncpu_local_model,
        )
        # Local TTS does not support streaming
        if self.config.stream:
            self.get_logger().warning(
                "Local TTS model does not support streaming. Setting stream to False."
            )
            self.config.stream = False
            self.inference_params = self.config.get_inference_params()

    def __get_audio_bytes(self, chunk: Union[bytes, str]) -> bytes:
        """Get audio bytes"""
        if isinstance(chunk, str):
            try:
                return base64.b64decode(chunk)
            except Exception as e:
                self.get_logger().error(f"Failed to decode base64 string: {e}")
                return b""
        else:
            return chunk

    def _stream_audio_udp(self):
        """Streams audio chunks to a target IP and port over UDP."""
        # import package
        try:
            from soundfile import SoundFile
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "UDP streaming for TextToSpeech component requires the soundfile module to be installed. Please install it with `pip install soundfile`"
            ) from e

        sock = None
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            while not self._stop_event.is_set():
                try:
                    # Wait for a new chunk with a timeout.
                    chunk = self._incoming_queue.get(
                        timeout=self.config.thread_shutdown_timeout
                    )
                    if chunk is None:
                        break
                except queue.Empty:
                    self.get_logger().debug(
                        f"No new audio for {self.config.thread_shutdown_timeout}s. "
                        "Gracefully shutting down UDP streaming thread."
                    )
                    break

                # change str to bytes if output is str
                output_bytes = self.__get_audio_bytes(chunk)
                if not output_bytes:
                    continue

                try:
                    with SoundFile(BytesIO(output_bytes)) as f:
                        # request float32 from soundfile
                        blocks = f.blocks(
                            self.config.block_size, dtype="float32", always_2d=True
                        )
                        for block in blocks:
                            if self._stop_event.is_set():
                                break
                            # Convert to raw bytes (float32 LE)
                            byte_block = block.tobytes()
                            sock.sendto(
                                byte_block,
                                (self.config.stream_to_ip, self.config.stream_to_port),
                            )
                except Exception as e:
                    self.get_logger().error(
                        f"Error processing of streaming audio chunk: {e}"
                    )
                    continue
        finally:
            self.get_logger().debug("Cleaning up UDP streaming resources.")
            if sock:
                sock.close()
            self.get_logger().debug("UDP streaming thread has finished.")

    def _playback_audio_local(self):
        """Play audio on local device using blocking writes.

        Uses PyAudio in blocking output mode — ``stream.write()`` blocks
        until the hardware is ready, providing natural back-pressure
        """
        try:
            import pyaudio
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "play_on_device local device configuration for TextToSpeech "
                "component requires soundfile and pyaudio modules to be "
                "installed. Please install them with `pip install soundfile pyaudio`"
            ) from e

        audio_interface = pyaudio.PyAudio()
        stream = None
        fmt = (0, 0)
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self._incoming_queue.get(
                        timeout=self.config.thread_shutdown_timeout
                    )
                    if chunk is None:
                        break
                except queue.Empty:
                    self.get_logger().debug(
                        f"No new audio for {self.config.thread_shutdown_timeout}s. "
                        "Gracefully shutting down playback thread."
                    )
                    break

                output_bytes = self.__get_audio_bytes(chunk)
                try:
                    stream, fmt = self.__play_chunk(
                        output_bytes, audio_interface, stream, fmt
                    )
                except Exception as e:
                    self.get_logger().error(f"Error processing audio chunk: {e}")
        finally:
            self.__cleanup_audio(stream, audio_interface)

    def __play_chunk(self, audio_bytes, audio_interface, stream, fmt):
        """Decode an audio chunk and write it to the PyAudio stream.

        Opens or recreates the stream if the audio format changed.

        :param audio_bytes: Raw audio file bytes (e.g. WAV)
        :param audio_interface: PyAudio instance
        :param stream: Current PyAudio stream or None
        :param fmt: Tuple of (sample_rate, channels) of the current stream
        :returns: Tuple of (stream, fmt) for reuse by the caller
        """
        from soundfile import SoundFile
        import pyaudio

        with SoundFile(BytesIO(audio_bytes)) as f:
            new_fmt = (f.samplerate, f.channels)
            if stream is None or new_fmt != fmt:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
                stream = audio_interface.open(
                    format=pyaudio.paFloat32,
                    channels=f.channels,
                    rate=f.samplerate,
                    output=True,
                    frames_per_buffer=self.config.block_size,
                    output_device_index=self.config.device,
                )
                fmt = new_fmt

            for block in f.blocks(
                self.config.block_size, dtype="float32", always_2d=True
            ):
                if self._stop_event.is_set():
                    break
                stream.write(block.tobytes())

        return stream, fmt

    @staticmethod
    def __cleanup_audio(stream, audio_interface):
        """Close PyAudio stream and terminate the audio interface."""
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if audio_interface:
            try:
                audio_interface.terminate()
            except Exception:
                pass

    def _play(self, audio_chunk: Union[bytes, str]):
        """
        Adds a chunk of audio data to the playback queue. If the playback thread is not running, it will be started automatically.
        """
        self._incoming_queue.put(audio_chunk)

        # If the playback thread doesn't exist or isn't alive, start it.
        with self._thread_lock:
            if self._playback_thread is None or not self._playback_thread.is_alive():
                self.get_logger().debug(
                    "Playback thread is not active. Starting a new one."
                )
                self._stop_event.clear()
                # Decide which worker to start based on config
                is_streaming_configured = (
                    self.config.stream_to_ip and self.config.stream_to_port
                )
                if is_streaming_configured:
                    target_func = self._stream_audio_udp
                    thread_name = "TTS-UDP-Streamer"
                    self.get_logger().debug(
                        "Playback thread is not active. Starting a new UDP streaming thread."
                    )
                else:
                    target_func = self._playback_audio_local
                    thread_name = "TTS-Local-Playback"
                    self.get_logger().debug(
                        "Playback thread is not active. Starting a new local playback thread."
                    )

                # Create and start thread
                self._playback_thread = threading.Thread(
                    target=target_func, name=thread_name, daemon=True
                )
                self._playback_thread.start()

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "stop_playback",
                "description": "Stop audio playback and clear any pending audio.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    def stop_playback(self, wait_for_thread: bool = True) -> bool:
        """
        Stops the playback thread and clears any pending audio.
        Can be used to interrupt the audio playback through an event.
        """
        self.get_logger().info("Stop requested. Signaling thread and clearing queue.")
        self._stop_event.set()

        # Clear the queue to ensure the thread doesn't process more items
        with self._incoming_queue.mutex:
            self._incoming_queue.queue.clear()

        # Add a sentinel value to unblock a waiting queue.get()
        self._incoming_queue.put(None)

        if wait_for_thread and self._playback_thread:
            self.get_logger().debug("Waiting for playback thread to terminate...")
            self._playback_thread.join()
            self.get_logger().debug("Thread terminated.")

        return True

    @component_action(
        description={
            "type": "function",
            "function": {
                "name": "say",
                "description": "Convert text to speech and play it on the robot's speaker to say something out loud.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to be spoken aloud.",
                        },
                    },
                    "required": ["text"],
                },
            },
        }
    )
    def say(self, text: str) -> bool:
        """
            Say the input text.

            This method converts the input text to speech and plays the speech on device if play_on_device is set to True and publishes to Audio topics if any publishers have been provided to the component. Any current playback is stopped.

            The method can be invoked as an action consequence of an event. For example, the robot can say 'I am low on battery" when a low battery event gets triggered.

        :param text: The text to be spoken out loud.
        :type text: str
        """
        try:
            # Stop current playback if active, so new speech isn't queued
            # behind stale audio
            if self._playback_thread and self._playback_thread.is_alive():
                self.stop_playback()
            self._execution_step(text=text)
            return True
        except Exception:
            return False

    def _handle_websocket_streaming(self) -> Optional[List]:
        """Handle streaming output from a websocket client"""
        try:
            tokens = self.resp_queue.get(block=True)
            if self.config.play_on_device:
                self._play(tokens)
            self._publish({"output": tokens})
        except Exception as e:
            self.get_logger().error(str(e))
            # raise a fallback trigger via health status
            self.health_status.set_fail_component()

    def _create_input(self, *_, **kwargs) -> Optional[Dict[str, Any]]:
        """Create inference input for TextToSpeech models
        :param args:
        :param kwargs:
        :rtype: dict[str, Any]
        """

        # set query as trigger
        if trigger := kwargs.get("topic"):
            query = self.trig_callbacks[trigger.name].get_output()
            if not query:
                return None
        elif text := kwargs.get("text"):
            query = text
        else:
            self.get_logger().error(
                "Trigger topic not found. TextToSpeech component needs to be given a valid trigger topic."
            )
            return None
        return {"query": query, **self.inference_params}

    def _execution_step(self, *args, **kwargs):
        """_execution_step.

        :param args:
        :param kwargs:
        """
        if self.run_type is ComponentRunType.EVENT and (trigger := kwargs.get("topic")):
            self.get_logger().debug(f"Received trigger on topic {trigger.name}")
        elif text := kwargs.get("text"):
            self.get_logger().debug(f"Received text: {text}")
        else:
            return None

        # create inference input
        inference_input = self._create_input(*args, **kwargs)
        # call model inference
        if not inference_input:
            self.get_logger().warning("Input not received, not calling model inference")
            return

        # conduct inference
        result = self._call_inference(inference_input)
        if result:
            if self.config.play_on_device:
                self._play(result["output"])

            # publish result
            self._publish(result)

    def _warmup(self):
        """Warm up and stat check"""

        inference_input = {
            "query": "Add the sum to the product of these three.",
            **self.inference_params,
        }

        # Run inference once to warm up and once to measure time
        if self.model_client:
            self.model_client.inference(inference_input)
        elif hasattr(self, "local_model"):
            self.local_model(inference_input)

        start_time = time.time()
        if self.model_client:
            self.model_client.inference(inference_input)
        elif hasattr(self, "local_model"):
            self.local_model(inference_input)
        elapsed_time = time.time() - start_time

        self.get_logger().warning(f"Approximate Inference time: {elapsed_time} seconds")
        self.get_logger().warning(
            f"RTF: {elapsed_time / 2}"  # approx audio length, 2 seconds
        )
