"""Design parameters are the "controllable" aspects of the design; these are what we
optimize when do design.
"""
from typing import List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt


# Define a generic type for a constraint
Constraint = Union[sciopt.LinearConstraint, sciopt.NonlinearConstraint]


class DesignParameters(object):
    """DesignParameters represents a vector of parameters over which the designer has
    some control. The design task involves changing these parameters to achieve good
    performance.

    Implemented as a generic vector of parameters with no bounds and no constraints.
    If bounds or constraints are needed, you should make a subclass for your design
    problem.
    """

    def __init__(
        self,
        size: int,
        names: Optional[list[str]] = None,
    ):
        """
        Initialize the DesignParameters object.

        args:
            size: the number of design variables
            names: a list of names for variables. If not provided, defaults to
                   "theta_0", "theta_1", ...
        """
        super(DesignParameters, self).__init__()
        self.size = size

        # Specify default behavior
        if names is None:
            names = [f"theta_{i}" for i in range(self.size)]
        self.names = names

        # Initialize parameter vector (try to respect bounds, but no guarantees that
        # we will respect constraints)
        self._values = jnp.zeros(self.size)
        for idx, bound in enumerate(self.bounds_list):
            lb, ub = bound
            if lb is not None and ub is not None:
                center = (lb + ub) / 2.0
            elif lb is not None:
                center = lb
            elif ub is not None:
                center = ub
            else:
                center = 0.0

            self._values = self._values.at[idx].set(center)

    def set_values(self, new_values: Union[jnp.ndarray, np.ndarray]):
        """Set the values of these design parameters using the given values.

        args:
            new_values: the array of new values
        """
        self._values = new_values

    def get_values(self) -> jnp.ndarray:
        """Return the values of these design parameters."""
        return self._values

    def get_values_np(self) -> np.ndarray:
        """Return the values of these design parameters."""
        return np.array(self._values)

    @property
    def bounds_list(self) -> Sequence[Tuple[Optional[float], Optional[float]]]:
        """Returns the bounds on the design parameters as a list. Each element
        of the list should be None (indicates no bound) or a tuple of (lower, upper)
        bounds.

        Default behavior (unless overridden by a subclass) is to not bound any variables
        """
        return [(None, None)] * self.size

    @property
    def constraints(self) -> List[Constraint]:
        """Returns a list of constraints, either `scipy.optimize.NonlinearConstraint` or
        `scipy.optimize.LinearConstraint` objects.

        Default (unless overridden by a subclass) is to not have any constraints
        """
        return []
