"""Event-driven motion alerts.

A MotionDetector component watches the camera stream and publishes:
- a Bool motion state topic, used here as an event source
- a Video message collecting the frames of each coherent motion sequence

When motion starts, an event triggers a TextToSpeech component that plays
an alert on the robot's speakers. The video of each motion episode is
published once the motion ends and can be recorded or passed to models
that accept image sequences.

The commented section at the bottom shows the point cloud variant, where
the centers of moving regions are published as coordinates instead of
videos.
"""

from agents.components import MotionDetector, TextToSpeech
from agents.config import MotionDetectorConfig, TextToSpeechConfig
from agents.clients import RoboMLWSClient
from agents.models import TransformersTTS
from agents.ros import Launcher, Topic, FixedInput, Event

# Define Topics
camera_image = Topic(name="/image_raw", msg_type="Image")
motion_state = Topic(name="/motion", msg_type="Bool")
motion_video = Topic(name="/motion_video", msg_type="Video")

# Setup the MotionDetector Component (The Watcher)
motion_config = MotionDetectorConfig(
    motion_estimation_func="frame_difference",
    threshold=0.5,
    motion_stop_delay=10,  # end a motion episode after 10 still frames
)

motion_detector = MotionDetector(
    inputs=[camera_image],
    outputs=[motion_state, motion_video],
    config=motion_config,
    trigger=camera_image,  # Runs on every frame
    component_name="motion_detector",
)

# Define the Event: fires when the motion state flips to True
event_motion_detected = Event(
    motion_state.msg.data == True,  # noqa: E712
    on_change=True,  # Trigger only on a state change to stop repeat triggering
)

# Setup the TextToSpeech Component (The Alarm)
# It has a fixed alert text and only runs when the motion event fires,
# playing the alert directly on the robot's speakers.
alert_text = FixedInput(
    name="alert",
    msg_type="String",
    fixed="Attention: motion detected in the monitored area.",
)

tts = TransformersTTS(name="tts")
roboml_tts = RoboMLWSClient(tts)

alert_speaker = TextToSpeech(
    inputs=[alert_text],
    trigger=event_motion_detected,  # Only runs when motion starts
    model_client=roboml_tts,
    config=TextToSpeechConfig(play_on_device=True),
    component_name="alert_speaker",
)

# --- Point cloud variant ---
# With a lidar/depth cloud input, the MotionDetector publishes the centers
# of moving regions as a PoseArray instead of videos. Providing the robot
# odometry topic as `position` enables ego-motion subtraction, so the
# component can be used while the robot is moving and the centers are
# published in the odometry frame:
#
# cloud = Topic(name="/points", msg_type="PointCloud2")
# odom = Topic(name="/odom", msg_type="Odometry")
# motion_centers = Topic(name="/motion_centers", msg_type="PoseArray")
# cloud_motion_detector = MotionDetector(
#     inputs=[cloud],
#     outputs=[motion_state, motion_centers],
#     config=MotionDetectorConfig(voxel_size=0.15, changed_voxel_threshold=10),
#     trigger=cloud,
#     position=odom,
#     component_name="motion_detector",
# )

# Launch
launcher = Launcher()
launcher.enable_ui(outputs=[motion_state, motion_video])
launcher.add_pkg(
    components=[motion_detector, alert_speaker],
    multiprocessing=True,
    package_name="automatika_embodied_agents",
)
launcher.bringup()
