from typing import List, Tuple, Optional

from architect.design import DesignParameters


class AGVDesignParameters(DesignParameters):
    """ExogenousParameters for the AGV localization task"""

    def __init__(self):
        """
        Initialize the design parameters for the AGV localization porblem.
        """
        names = [
            "initial_x",
            "initial_y",
            "beacon_1_x",
            "beacon_1_y",
            "beacon_2_x",
            "beacon_2_x",
        ]
        super(AGVDesignParameters, self).__init__(6, names)

    @property
    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """Returns the bounds on the design parameters as a list. Each element
        of the list should be None (indicates no bound) or a tuple of (lower, upper)
        bounds.

        Default behavior (unless overridden by a subclass) is to not bound any variables
        """
        return [
            (None, None),  # no bounds on control gains
            (None, None),  # no bounds on control gains
            (-3.0, 0.0),  # beacons should be in this box
            (-1.0, 1.0),
            (-3.0, 0.0),
            (-1.0, 1.0),
        ]
