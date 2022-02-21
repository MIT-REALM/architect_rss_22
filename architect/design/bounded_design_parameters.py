"""Design parameters are the "controllable" aspects of the design; these are what we
optimize when do design.
"""
from typing import List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import scipy.optimize as sciopt

from .design_parameters import DesignParameters


# Define a generic type for a constraint
Constraint = Union[sciopt.LinearConstraint, sciopt.NonlinearConstraint]


class BoundedDesignParameters(DesignParameters):
    """BoundedDesignParameters represents a set of design parameters with non-optional
    bounds.
    """

    def __init__(
        self,
        size: int,
        bounds: jnp.ndarray,
        names: Optional[list[str]] = None,
    ):
        """
        Initialize the BoundedDesignParameters object.

        args:
            size: the number of design variables
            bounds: a (size, 2) array of upper and lower bounds for each parameter.
            names: a list of names for variables. If not provided, defaults to
                   "theta_0", "theta_1", ...
        """
        self.bounds = bounds
        super(BoundedDesignParameters, self).__init__(size, names)

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
    def bounds_list(self) -> Sequence[Tuple[float, float]]:
        """Returns the bounds on the design parameters as a list. Each element
        of the list should be a tuple of (lower, upper) bounds.
        """
        return [(lb.item(), ub.item()) for lb, ub in self.bounds]

    @property
    def constraints(self) -> List[Constraint]:
        """No constraints other than the bounds"""
        return []
