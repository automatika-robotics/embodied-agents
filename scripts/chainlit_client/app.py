from io import BytesIO
from typing import Union, Optional, List
from enum import Enum

import chainlit as cl
from chainlit.element import ElementBased
from chainlit.input_widget import TextInput

import rclpy
from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, String


class Status(Enum):
    INIT = 0
    RECEIVED_TEXT = 1
    RECEIVED_AUDIO = 2
    TIMEOUT = 3


class ClientNode(Node):
    """
    Cli based text client with a publisher and subscriber.
    """

    def __init__(self) -> None:
        """
        Constructs a new instance.
        """
        super().__init__("cli_client")
        self.msg: Optional[Union[str, bytes]] = None
        # Start with defaults
        self.set_trigger("text0", "audio0")
        self.set_target("text1", "audio1")

    def publish(self, prompt: Union[str, bytes]) -> None:
        """
        Publish to the trigger topics and listen to the target topics

        :param      prompt:  The prompt/question
        :type       prompt:  {str, bytes}

        :returns:   None
        :rtype:     None
        """

        # set timeout flag
        self.msg_received = Status.INIT
        # Check for publishers on available topic and quit if none available
        if isinstance(prompt, bytes):
            if not self.count_subscribers(self.audio_trigger) > 0:
                self.get_logger().info(
                    f"No one is listening to {self.audio_trigger}, so I am timing out"
                )
                self.timer = self.create_timer(0, self.timer_callback)
                return None
            msg = ByteMultiArray()
            msg.data = prompt
            self.audio_publisher.publish(msg)
            self.get_logger().info(f"Publishing to {self.audio_trigger}")
        else:
            if not self.count_subscribers(self.text_trigger) > 0:
                self.get_logger().info(
                    f"No one is listening to {self.text_trigger}, so I am timing out"
                )
                self.timer = self.create_timer(0, self.timer_callback)
                return None
            # Create and publish message
            msg = String()
            msg.data = prompt
            self.text_publisher.publish(msg)
            self.get_logger().info(f"Publishing to {self.text_trigger}")

        self.get_logger().info("Now listening..")

    def listener_callback(self, msg: Union[String, ByteMultiArray]) -> None:
        """
        Listener callback

        :param      msg:  The message
        :type       msg:  {ROS Message}
        """
        if isinstance(msg, String):
            self.msg_received = Status.RECEIVED_TEXT
            self.get_logger().info(f"A: {msg.data}")
            self.msg = msg.data
        elif isinstance(msg, ByteMultiArray):
            self.msg_received = Status.RECEIVED_AUDIO
            self.get_logger().info("A: Audio bytes")
            self.msg = b"".join(msg.data)
        else:
            self.get_logger().error(
                "Something went wrong. Received message is neither String nor ByteMultiArray"
            )

    def timer_callback(self):
        """
        Timer Callback just for destroying the time and end node spin_once
        """
        # the timer should be destroyed once utilized
        self.destroy_timer(self.timer)
        self.msg_received = Status.TIMEOUT

    def set_trigger(self, text_trigger: str, audio_trigger: str):
        """
        Set topic to send messages to
        """
        if hasattr(self, "text_publisher"):
            self.destroy_publisher(self.text_publisher)
        self.text_trigger = text_trigger
        self.text_publisher = self.create_publisher(String, self.text_trigger, 1)

        if hasattr(self, "audio_publisher"):
            self.destroy_publisher(self.audio_publisher)
        self.audio_trigger = audio_trigger
        self.audio_publisher = self.create_publisher(
            ByteMultiArray, self.audio_trigger, 1
        )

    def set_target(self, text_target: str, audio_target: str):
        """
        Set topic to receive messages from
        """
        if hasattr(self, "text_subscription"):
            self.destroy_subscription(self.text_subscription)
        self.text_target = text_target
        self.text_subscription = self.create_subscription(
            String, self.text_target, self.listener_callback, 1
        )

        if hasattr(self, "audio_subscription"):
            self.destroy_subscription(self.audio_subscription)
        self.audio_target = audio_target
        self.audio_subscription = self.create_subscription(
            ByteMultiArray, self.audio_target, self.listener_callback, 1
        )


@cl.on_chat_start
async def on_chat_start():
    """
    On chat start, specify default settings
    """
    # Init rclpy
    if not rclpy.ok():
        rclpy.init()
    await cl.ChatSettings([
        TextInput(
            id="text_trigger",
            label="String topic to send message to",
            initial="text0",
        ),
        TextInput(
            id="text_target",
            label="String topic to listen to for response",
            initial="text1",
        ),
        TextInput(
            id="audio_trigger",
            label="Audio topic to send message to",
            initial="audio0",
        ),
        TextInput(
            id="audio_target",
            label="Audio topic to listen to for response",
            initial="audio1",
        ),
        TextInput(id="timeout", label="Timeout (sec)", initial="30"),
    ]).send()
    cl.user_session.set("timeout", 30)
    client: ClientNode = ClientNode()
    cl.user_session.set("client", client)
    await cl.Message(
        content="Welcome to Leibniz ROS client. Set the input/output topics in settings. Then type your message or press `P` to send audio!"
    ).send()


@cl.on_settings_update
async def setup_ros_node(settings):
    """
    On settings update, update nodes
    """
    client: ClientNode = cl.user_session.get("client")
    client.set_trigger(settings["text_trigger"], settings["audio_trigger"])
    client.set_target(settings["text_target"], settings["audio_target"])
    if not settings["timeout"].isdigit():
        return
    cl.user_session.set("timeout", int(settings["timeout"]))


@cl.step(type="run")
def publish_on_ros(msg: Union[str, bytes]):
    """Publish input to the ROS Client node.
    :param msg:
    :type msg: Union[str, bytes]
    """
    timeout: int = cl.user_session.get("timeout")
    client: ClientNode = cl.user_session.get("client")
    client.publish(msg)
    rclpy.spin_once(client, timeout_sec=timeout)


@cl.step(type="run")
async def handle_output(msg_type: type):
    """Handle Output from the ROS Client node.
    :param msg_type:
    :type msg_type: type
    """
    client: ClientNode = cl.user_session.get("client")
    if client.msg_received is Status.INIT:
        await cl.Message(
            content=f"I did not receive a message on **{client.text_target}** or **{client.audio_target}**. Timedout.",
        ).send()
    elif client.msg_received is Status.RECEIVED_TEXT:
        await cl.Message(
            content=f"{client.msg}",
        ).send()
    elif client.msg_received is Status.RECEIVED_AUDIO:
        output_audio_el = cl.Audio(content=client.msg, name="Response Audio")
        await cl.Message(
            author="Robot",
            type="assistant_message",
            content="",
            elements=[output_audio_el],
        ).send()
    else:
        trig = client.audio_trigger if msg_type is bytes else client.text_trigger
        await cl.Message(
            content=f"There is no one listening on **{trig}**. Is this the correct topic. If not, set the correct trigger and response topics in the settings.",
        ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    """
    On message, handle text message
    """
    publish_on_ros(msg.content)
    await handle_output(type(msg))


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Receive audio chunks
    :param chunk:
    :type chunk: cl.AudioChunk
    """
    if chunk.isStart:
        # Initialize new audio buffer
        buffer = BytesIO()
        buffer.name = "input_audio"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # write chunks to buffer
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: List[ElementBased]):
    """Publish audio to the topic.
    :param elements:
    :type elements: list[ElementBased]
    """
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    audio_bytes = audio_buffer.read()

    # Add users audio to the chat
    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_bytes, name="User Audio"
    )
    await cl.Message(
        author="User",
        type="user_message",
        content="",
        elements=[input_audio_el, *elements],
    ).send()

    # publish using ROS client
    publish_on_ros(audio_bytes)
    await handle_output(type(audio_bytes))


@cl.on_chat_end
async def on_chat_end():
    """
    On chat end destroy client nodes
    """
    if rclpy.ok():
        client: ClientNode = cl.user_session.get("client")
        client.destroy_node()
        rclpy.shutdown()
