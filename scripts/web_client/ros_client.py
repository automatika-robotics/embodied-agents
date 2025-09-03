import asyncio
import base64
import cv2
from rclpy.node import Node
from std_msgs.msg import String, ByteMultiArray
from sensor_msgs.msg import CompressedImage, Image
from automatika_embodied_agents.msg import StreamingString
from typing import Union, Callable
from ros_sugar.io.utils import read_compressed_image, image_pre_processing


class ClientNode(Node):
    """
    ROS2 node to handle communication between the web client and the EmbodiedAgents system.
    """

    def __init__(self, websocket_callback: Callable):
        """
        Constructs a new instance.
        """
        super().__init__("web_client_node")
        self.websocket_callback = websocket_callback
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # Initialize placeholders
        self.text_subscription = None
        self.string_subscription = None
        self.image_subscription = None
        self.video_stream_active = False

        self.set_topics(
            text_trigger="text0",
            text_target="text1",
            audio_trigger="audio0",
            audio_target="audio1",
            video_stream_topic="image_raw",
            enable_streaming=False,
        )
        self.create_timer(2.0, self.check_video_publisher)

    def publish_text(self, prompt: str):
        """Publish text to the trigger topic after checking for subscribers."""
        if self.count_subscribers(self.text_trigger) == 0:
            error_msg = f'Error: No subscribers found for the topic "{self.text_trigger}". Please check the topic name in settings.'
            self.get_logger().error(error_msg)
            payload = {"type": "error", "payload": error_msg}
            asyncio.run_coroutine_threadsafe(
                self.websocket_callback(payload), self.loop
            )
            return

        msg = String()
        msg.data = prompt
        self.text_publisher.publish(msg)
        self.get_logger().info(f'Publishing to topic "{self.text_trigger}": "{prompt}"')

    def publish_audio(self, audio_bytes: bytes):
        """Publish audio to the trigger topic after checking for subscribers."""
        if self.count_subscribers(self.audio_trigger) == 0:
            error_msg = f'Error: No subscribers found for the topic "{self.audio_trigger}". Please check the topic name in settings.'
            self.get_logger().error(error_msg)
            payload = {"type": "error", "payload": error_msg}
            asyncio.run_coroutine_threadsafe(
                self.websocket_callback(payload), self.loop
            )
            return

        msg = ByteMultiArray()
        try:
            msg.data = [bytes([b]) for b in audio_bytes]

            self.audio_publisher.publish(msg)
            self.get_logger().info(
                f'Successfully published {len(audio_bytes)} audio bytes to topic "{self.audio_trigger}"'
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to create or publish ByteMultiArray message: {e}"
            )

    def listener_callback(self, msg: Union[StreamingString, String, ByteMultiArray]):
        """Callback for received ROS messages."""
        if isinstance(msg, String):
            self.get_logger().info(
                f'Received string from "{self.text_target}": "{msg.data}"'
            )
            payload = {"type": "text", "payload": msg.data}
        elif isinstance(msg, StreamingString):
            self.get_logger().info(
                f'Received streaming string from "{self.text_target}"'
            )
            payload = {"type": "stream", "payload": msg.data, "done": msg.done}
        elif isinstance(msg, ByteMultiArray):
            self.get_logger().info(
                f'Received {len(msg.data)} audio bytes from "{self.audio_target}"'
            )
            # Encode audio bytes to base64 to send as a JSON string
            encoded_audio = base64.b64encode(b"".join(msg.data)).decode("utf-8")
            payload = {"type": "audio", "payload": encoded_audio}
        else:
            self.get_logger().error("Received message of unknown type.")
            return

        # Use the running asyncio loop to call the async websocket_callback
        asyncio.run_coroutine_threadsafe(self.websocket_callback(payload), self.loop)

    def _process_and_send_image(self, cv_image):
        try:
            # Encode image as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, buffer = cv2.imencode(".jpg", cv_image, encode_param)
            if not result:
                self.get_logger().error("Failed to encode image to JPEG format.")
                return

            # Convert to base64
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            payload = {"type": "video_frame", "payload": jpg_as_text}

            asyncio.run_coroutine_threadsafe(
                self.websocket_callback(payload), self.loop
            )

        except Exception as e:
            self.get_logger().error(f"Failed to encode and send image frame: {e}")

    def image_callback(self, msg: Union[Image, CompressedImage]):
        try:
            if isinstance(msg, Image):
                cv_image = image_pre_processing(msg)
            else:
                cv_image = read_compressed_image(msg)

            if cv_image is not None:
                self._process_and_send_image(cv_image)
            else:
                self.get_logger().warn("read_compressed_image returned None.")
        except Exception as e:
            self.get_logger().error(f"Failed to process CompressedImage message: {e}")

    def check_video_publisher(self):
        """Check for subscribers to video topics"""
        is_publisher_present = self.count_publishers(self.video_stream_topic) > 0

        if is_publisher_present and not self.video_stream_active:
            self.get_logger().info(
                f"Video stream started on topic '{self.video_stream_topic}'."
            )
            self.video_stream_active = True
            payload = {"type": "video_stream_start"}
            asyncio.run_coroutine_threadsafe(
                self.websocket_callback(payload), self.loop
            )
        elif not is_publisher_present and self.video_stream_active:
            self.get_logger().info(
                f"Video stream stopped on topic '{self.video_stream_topic}'."
            )
            self.video_stream_active = False
            payload = {"type": "video_stream_stop"}
            asyncio.run_coroutine_threadsafe(
                self.websocket_callback(payload), self.loop
            )

    def set_topics(
        self,
        text_trigger: str,
        text_target: str,
        audio_trigger: str,
        audio_target: str,
        video_stream_topic: str,
        enable_streaming: bool,
    ):
        """Set up or update publishers and subscribers for the given topics."""
        self.text_trigger = text_trigger
        self.text_target = text_target
        self.audio_trigger = audio_trigger
        self.audio_target = audio_target
        self.video_stream_topic = video_stream_topic

        # Text publisher
        if hasattr(self, "text_publisher"):
            self.destroy_publisher(self.text_publisher)
        self.text_publisher = self.create_publisher(String, self.text_trigger, 10)

        # Audio publisher
        if hasattr(self, "audio_publisher"):
            self.destroy_publisher(self.audio_publisher)
        self.audio_publisher = self.create_publisher(
            ByteMultiArray, self.audio_trigger, 10
        )

        # Destroy old text subscriptions before creating new ones
        if self.text_subscription:
            self.destroy_subscription(self.text_subscription)
            self.text_subscription = None
        if self.string_subscription:
            self.destroy_subscription(self.string_subscription)
            self.string_subscription = None

        # Create the correct text subscriber based on the streaming flag
        if enable_streaming:
            self.get_logger().info(
                f'Creating subscriber for StreamingString on "{self.text_target}"'
            )
            self.text_subscription = self.create_subscription(
                StreamingString, self.text_target, self.listener_callback, 10
            )
        else:
            self.get_logger().info(
                f'Creating subscriber for String on "{self.text_target}"'
            )
            self.string_subscription = self.create_subscription(
                String, self.text_target, self.listener_callback, 10
            )

        # Audio subscriber
        if hasattr(self, "audio_subscription"):
            self.destroy_subscription(self.audio_subscription)
        self.audio_subscription = self.create_subscription(
            ByteMultiArray, self.audio_target, self.listener_callback, 10
        )

        if self.image_subscription:
            self.destroy_subscription(self.image_subscription)

        # Create Image or Compressed image listener based on topic name
        if "compressed" in self.video_stream_topic:
            self.image_subscription = self.create_subscription(
                CompressedImage,
                self.video_stream_topic,
                self.image_callback,
                10,
            )
        else:
            self.image_subscription = self.create_subscription(
                Image,
                self.video_stream_topic,
                self.image_callback,
                10,
            )

        self.video_stream_active = False

        self.get_logger().info("ROS topics configured:")
        self.get_logger().info(f"  - Text Output (Pub): {self.text_trigger}")
        self.get_logger().info(
            f"  - Text Input (Sub): {self.text_target} (Streaming: {enable_streaming})"
        )
        self.get_logger().info(f"  - Audio Output (Pub): {self.audio_trigger}")
        self.get_logger().info(f"  - Audio Input (Sub): {self.audio_target}")
        self.get_logger().info(f"  - Video Input (Sub): {self.video_stream_topic}")
