from typing import List, Dict, Literal, Optional, Iterable, Tuple
from attr import define, field
import numpy as np

from rclpy.logging import get_logger

from .utils import _read_spec_file


def _size_validator(instance, attribute, value):
    """Size validator for positions, velocities, accelerations or efforts"""
    if value.size > 0 and value.size != len(instance.joints_names):
        raise ValueError(
            f"Length of {attribute} must be the same as length of joint_names: {len(value)} not equal to {len(instance.joints_names)}"
        )


@define(kw_only=True)
class JointsData:
    joints_names: List[str] = field()
    positions: np.ndarray = field(
        default=np.array([], dtype=np.float64), validator=_size_validator
    )
    velocities: np.ndarray = field(
        default=np.array([], dtype=np.float64), validator=_size_validator
    )
    accelerations: np.ndarray = field(
        default=np.array([], dtype=np.float64), validator=_size_validator
    )
    efforts: np.ndarray = field(
        default=np.array([], dtype=np.float64), validator=_size_validator
    )
    duration: float = field(default=0.0)
    delay: float = field(default=0.0)

    def __attrs_post_init__(self):
        """Last sanity check"""
        if (
            self.positions.size == 0
            and self.velocities.size == 0
            and self.accelerations.size == 0
            and self.efforts.size == 0
        ):
            raise ValueError(
                "JointsData should contain at least one of the following: positions, velocities, accelerations, efforts"
            )

    def get_mapped_state(
        self,
        state_type: Literal["positions", "velocities", "accelerations", "efforts"],
        joint_names_map: Dict[str, str],
    ) -> Optional[Dict[str, np.float32]]:
        """Return the state mapped according to the dataset keys provided in the config"""
        # Get particular state values of the type required
        state_values = getattr(self, state_type, None)

        if state_values is None or state_values.size == 0:
            return None

        # Build a name -> index lookup table
        name_to_index = {name: i for i, name in enumerate(self.joints_names)}

        mapped = {}

        for target_name, source_name in joint_names_map.items():
            idx = name_to_index.get(source_name)
            if idx is None:
                return None
            mapped[target_name] = state_values[idx]

        return mapped


# Map requirement categories -> URDF limit keys
req_map = {
    "positions": ["lower", "upper"],
    "efforts": ["effort"],
    "velocities": ["velocity"],
    "accelerations": ["velocity"],
}


def check_joint_limits(
    joints_limits: Dict[str, Optional[Dict[str, float]]], requirements: Iterable[str]
) -> Tuple:
    """
    Validate that required joint limits are provided for all joints.

    :param joints_limits: Output of parse_urdf_joints().
    :type joints_limits: Dict[str, Optional[Dict[str, float]]]

    :param requirements: Iterable of requirement categories:
                         - "positions"  -> requires "lower" and "upper"
                         - "efforts"    -> requires "effort"
                         - "velocities" or "accelerations" -> requires "velocity"
    :type requirements: Iterable[str]

    :return: A tuple containing a boolean indicating if all requirements are satisfied,
             and a list of human-readable error messages.
    :rtype: Tuple[bool, List[str]]
    """

    # Build final required keys
    required_keys = []
    for req in requirements:
        required_keys.extend(req_map.get(req, []))

    errors = []

    for joint, limits in joints_limits.items():
        if limits is None:
            # No limits available at all
            missing = required_keys
            if missing:
                errors.append(
                    f"Joint '{joint}' has no <limit> tag but requires: {missing}"
                )
            continue

        # Check required keys for this joint
        for key in required_keys:
            if limits.get(key) is None:
                errors.append(f"Joint '{joint}' missing required limit '{key}'.")

    return (len(errors) == 0, errors)


def cap_actions_with_limits(
    joint_names: List[str],
    target_actions: np.ndarray,
    limits_dict: Optional[Dict[str, Optional[Dict[str, float]]]],
    action_type: Literal["positions", "velocities", "efforts", "accelerations"],
    logger_name: str,
) -> np.ndarray:
    """
    Cap the target actions for a set of joints based on their specified limits.

    :param joint_names: A list of names corresponding to the joints.
    :type joint_names: List[str]
    :param target_actions:
        The target actions (positions, velocities, efforts, or accelerations) for each joint.
    :type target_actions: np.ndarray(dtype=np.float32)
    :param limits_dict:
        A dictionary containing the limit specifications for each joint. Each key is a joint name,
        and the value is another dictionary with keys 'lower' and/or 'upper' (for positions)
        or 'velocity', 'effort', etc. (for other action types).
    :type limits_dict: Dict[str, Optional[Dict[str, float]]]
    :param action_type: The type of actions being capped.
    :type action_type: Literal["positions", "velocities", "efforts", "accelerations"]
    :param logger_name:
        The name of the component to use for logging warnings.
    :type logger_name: str

    :return:
        An np.array of capped actions for each joint, ensuring they do not exceed their specified limits.
    :rtype: np.ndarray(dtype=np.float32)

    Notes
    -----
    - For positions, both 'lower' and 'upper' limits must be provided in `limits_dict`.
    - For velocities, efforts, and accelerations, the corresponding limit key (e.g., 'velocity') must be provided.
    """

    result = np.copy(target_actions)
    required_limits = req_map[action_type]

    # TODO: Implement correct limit checking for acceleration based on velocities and timestep
    # Special case: accelerations -> skip limit checking entirely
    if action_type == "accelerations":
        get_logger(logger_name).warning(
            "Acceleration action is not currently limit-checked "
        )
        return target_actions

    for idx, (jname, target) in enumerate(
        zip(joint_names, target_actions, strict=True)
    ):
        # If limit missing in dict
        joint_limits = limits_dict.get(jname)
        if joint_limits is None:
            get_logger(logger_name).warning(
                f"Joint '{jname}' has no limit assigned — action left uncapped."
            )
            result[idx] = target
            continue

        # Check required limits exist
        missing_any = False
        for limit_key in required_limits:
            if joint_limits.get(limit_key) is None:
                get_logger(logger_name).warning(
                    f"Joint '{jname}' missing required limit '{limit_key}' — action left uncapped."
                )
                missing_any = True
        if missing_any:
            result[idx] = target
            continue

        # Safety capping logic
        capped = target
        if action_type == "positions":
            lower = joint_limits["lower"]
            upper = joint_limits["upper"]

            if capped < lower:
                get_logger(logger_name).warning(
                    f"Position for joint '{jname}' capped: {capped} -> {lower}"
                )
                capped = lower
            elif capped > upper:
                get_logger(logger_name).warning(
                    f"Position for joint '{jname}' capped: {capped} -> {upper}"
                )
                capped = upper

        else:
            # effort / velocity / acceleration -> absolute magnitude clamp
            limit_key = required_limits[0]
            max_mag = joint_limits[limit_key]

            if abs(capped) > max_mag:
                new_val = max_mag if capped > 0 else -max_mag
                get_logger(logger_name).warning(
                    f"{action_type.capitalize()} for joint '{jname}' capped: "
                    f"{capped} -> {new_val}"
                )
                capped = new_val

        result[idx] = capped

    return result


def parse_urdf_joints(path_or_url: str) -> Dict:
    """
    Parse a URDF file and extract joint limits.

    :param path_or_url: The file path or URL of the URDF file to parse.
    :type path_or_url: str
    :return: A dictionary where the keys are joint names and the values are dictionaries containing the joint limits (lower, upper, effort, velocity).
             If no limits are found for a joint, the value will be `None`.
    :rtype: dict
    """
    root = _read_spec_file(path_or_url, spec_type="xml")

    joints_limits = {}

    for joint in root.findall("joint"):
        name = joint.get("name")

        limit_tag = joint.find("limit")
        if limit_tag is None:
            joints_limits[name] = None
            continue

        # Extract attributes if present
        limits = {}
        for attr in ["lower", "upper", "effort", "velocity"]:
            value = limit_tag.get(attr)
            limits[attr] = float(value) if value is not None else None

        joints_limits[name] = limits

    return joints_limits
