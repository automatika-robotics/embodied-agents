from typing import List, Dict, Literal, Optional, Iterable, Tuple
from attr import define, field
import numpy as np

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


def find_missing_values(check_list, string_list: List) -> List:
    """
    Return strings from `string_list` that do NOT appear in the dictionary's values.
    """
    # Convert values to a set for efficient lookup
    value_set = set(check_list)
    # Collect strings that are not found among the values
    missing = [s for s in string_list if s not in value_set]

    return missing


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

    # Map requirement categories -> URDF limit keys
    req_map = {
        "positions": ["lower", "upper"],
        "efforts": ["effort"],
        "velocities": ["velocity"],
        "accelerations": ["velocity"],
    }

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


def parse_urdf_joints(path_or_url: str) -> Dict:
    """
    Parse a URDF file and extract joint limits.
    Returns:
        dict: {joint_name: {lower, upper, effort, velocity} or None}
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
