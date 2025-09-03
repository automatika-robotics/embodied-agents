from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import logging
import rclpy
import threading
import asyncio
from contextlib import asynccontextmanager
from ros_client import ClientNode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ROS2 in the main thread
rclpy.init()

# Global variable to hold the ROS node
ros_node = None


def spin_ros():
    """Function to spin the ROS node, to be run in a separate thread."""
    global ros_node
    if ros_node:
        try:
            rclpy.spin(ros_node)
        except rclpy.executors.ExternalShutdownException:
            # This is expected when rclpy.shutdown() is called
            pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Manages the startup and shutdown of the ROS2 node.
    """
    global ros_node, ros_thread
    # A dummy callback until a real websocket is connected
    ros_node = ClientNode(websocket_callback=lambda _: asyncio.sleep(0))
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()
    logger.info("ROS2 node started and spinning in a background thread.")

    yield  # The application runs after this point

    # --- Shutdown Logic ---
    logger.info("Shutting down ROS2 node...")
    if ros_node:
        ros_node.destroy_node()
    rclpy.shutdown()
    logger.info("ROS2 node shut down completely.")


app = FastAPI(lifespan=lifespan)

# Mount the 'static' directory to serve CSS and JS files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def get_index():
    """Serves the main index.html file."""
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection for the chat interface."""
    await websocket.accept()
    logger.info("WebSocket connection established.")

    # Create an async function to send messages to the websocket
    async def send_to_websocket(data):
        await websocket.send_json(data)

    # Update the node's callback to use the new websocket connection
    global ros_node
    if ros_node:
        ros_node.websocket_callback = send_to_websocket

    try:
        while True:
            # Wait for a message from the client
            data = await websocket.receive_json()

            # Pass the message to the ROS node
            if ros_node:
                if data["type"] == "text":
                    logger.info(f"Received message: {data}")
                    ros_node.publish_text(data["payload"])
                elif data["type"] == "audio":
                    logger.info("Received message: audio bytes")
                    import base64

                    audio_bytes = base64.b64decode(data["payload"])
                    ros_node.publish_audio(audio_bytes)
                elif data["type"] == "settings":
                    logger.info(f"Received message: {data}")
                    ros_node.set_topics(**data["payload"])

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"An error occurred in the WebSocket: {e}")
