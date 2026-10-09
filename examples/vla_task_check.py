from agents.components import VLA, DecisionComponent
from agents.config import VLAConfig
from agents.clients import LeRobotClient, GenericHTTPClient
from agents.models import LeRobotPolicy, GenericDecisionModel
from agents.ros import Topic, Launcher, Event

# Create the topics
state = Topic(name="/isaac_joint_states", msg_type="JointState")  # robot state
camera1 = Topic(name="/front_camera/image_raw", msg_type="Image")  # Camera 1
camera2 = Topic(name="/wrist_camera/image_raw", msg_type="Image")  # Camera 2
joints_action = Topic(name="/isaac_joint_command", msg_type="JointState")


# --- Create the VLA component --- #
# Specify LeRobot Policy to use
policy = LeRobotPolicy(
    name="my_policy",
    policy_type="smolvla",
    checkpoint="aleph-ra/smolvla_finetune_pick_orange_20000",
    dataset_info_file="https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange/resolve/main/meta/info.json",
)

# Create the client for LeRobot Policy Server
client = LeRobotClient(model=policy)

# joint names map (dataset_names in info.json -> robot_names in the urdf)
joints_map = {
    "shoulder_pan.pos": "Rotation",
    "shoulder_lift.pos": "Pitch",
    "elbow_flex.pos": "Elbow",
    "wrist_flex.pos": "Wrist_Pitch",
    "wrist_roll.pos": "Wrist_Roll",
    "gripper.pos": "Jaw",
}

# camera inputs map (dataset_names in info.json -> image topics)
camera_map = {"front": camera1, "wrist": camera2}

config = VLAConfig(
    observation_sending_rate=3,
    action_sending_rate=3,
    joint_names_map=joints_map,
    camera_inputs_map=camera_map,
    robot_urdf_file="https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/refs/heads/main/Simulation/SO101/so101_new_calib.urdf",
)

vla = VLA(
    inputs=[state, camera1, camera2],
    outputs=[joints_action],
    model_client=client,
    config=config,
    component_name="vla_with_smolvla",
)


# --- Create a decision component that checks whether the task is done --- #
# A decision model answers typed questions in one forward pass, without
# generating text. Checking a camera image needs one that reads images, such as
# OpenJev (note its non-commercial license), served by llama.cpp's llama-server:
#   llama-server -m OpenJev-Q4_K_M.gguf --mmproj mmproj-OpenJev-Q8_0.gguf --alias openjev --port 8090
decision_client = GenericHTTPClient(
    GenericDecisionModel(name="openjev", checkpoint="openjev"), port=8090
)

# The checker looks at the front camera once a second. The task it checks is
# not written here: it asks the VLA for the task of the goal that is running,
# through the VLA's get_current_task action, so the same recipe works for any
# task sent to the VLA. While no goal is running, the action reports so and the
# checker asks nothing.
task_checker = DecisionComponent(
    inputs=[camera1],
    action_states={"task": vla.get_current_task},
    questions={
        "done": {
            "type": "noul",  # a yes/no question, answered with the probability of yes
            "instructions": "Has the robot completed the task given in the state?",
        }
    },
    model_client=decision_client,
    trigger=1.0,
    component_name="task_checker",
)

# Each question is answered on its own topic, <component_name>/<question_id>
task_done = Topic(name="task_checker/done", msg_type="Decision")

# End the goal when the checker is sure enough that the task is done, and after
# 400 timesteps otherwise. The threshold depends on how the task is worded: 0.8
# separated finished from unfinished scenes on recorded frames of this task
# ('Grab orange and place into plate'), and needs tuning for other tasks.
vla.set_termination_trigger(
    mode="event",
    stop_event=Event(task_done.msg.noul > 0.8),
    max_timesteps=400,
)

# --- Launch the components --- #
# A task can then be sent to the VLA as an action goal, e.g. from the command line:
#   ros2 action send_goal /vla_with_smolvla/manipulate_with_vla automatika_embodied_agents/action/VisionLanguageAction "{task: 'Grab orange and place into plate'}"
launcher = Launcher()
launcher.add_pkg(
    components=[vla, task_checker],
    package_name="automatika_embodied_agents",
    multiprocessing=True,
)
launcher.on_process_fail()
launcher.fallback_rate = 1 / 10  # 0.1 Hz or 10 seconds
launcher.bringup()
