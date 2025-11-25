from typing import List, Optional
from attr import define, field
import numpy as np


@define(kw_only=True)
class JointsData:
    joints_names : List[str] = field()
    positions : np.ndarray = field()
    velocities: Optional[np.ndarray] = field(default=np.array([], dtype=np.float64))
    accelerations: Optional[np.ndarray] = field(default=np.array([], dtype=np.float64))
    efforts: Optional[np.ndarray] = field(default=np.array([], dtype=np.float64))
    duration : float = field(default=0.0)
    delay : float = field(default=0.0)
