"""Design parameters are the "controllable" aspects of the design; these are what we
optimize when do design.
"""
from typing import List, Tuple, Optional, Union

import jax.numpy as jnp
import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

# Define some types we'll use later
GenericConstraint = Union[LinearConstraint, NonlinearConstraint]
Bound = Tuple[float, float]


class DesignParameters(object):
    """DesignParameters represents a vector of parameters over which the designer has
    some control. The design task involves changing these parameters to achieve good
    performance.
    """

    def __init__(
        self,
        num_vars: int,
        names: Optional[List[str]] = None,
        bounds: Optional[List[Optional[Bound]]] = None,
        constraints: Optional[GenericConstraint] = None,
    ):
        """
        Initialize the DesignParameters object.

        args:
            num_vars: the number of design variables
            names: a list of names for variables. If not provided, defaults to
                   "theta_0", "theta_1", ...
            bounds: a list of Tuples specifying (lower, upper) bounds on the
                    design parameters. Any element can be None to indicate that the
                    corresponding parameter is unbounded. If not provided, defaults to
                    no bounds on any parameter.
            constraints: a list of constraints, either
                         `scipy.optimize.NonlinearConstraint` or
                         `scipy.optimize.LinearConstraint` objects. If not provided,
                         defaults to no constraints.
        """
        super(DesignParameters, self).__init__()
        self.num_vars = num_vars

        # Specify default behavior
        if names is None:
            names = [f"theta_{i}" for i in range(self.num_vars)]
        self.names = names

        if bounds is None:
            bounds = [None for _ in range(self.num_vars)]
        self.bounds = bounds

        if constraints is None:
            constraints = []
        self.constraints = constraints

        # Initialize parameter vector (try to respect bounds, but no guarantees that
        # we will respect constraints)
        self._values = jnp.zeros(self.num_vars)
        for idx, bound in enumerate(self.bounds):
            if bound is not None:
                lb, ub = bound
                center = (lb + ub) / 2.0
                self._values = self._values.at[idx].set(center)

    def set_values(self, new_values: Union[jnp.ndarray, np.ndarray]):
        """Set the values of these design parameters using the given values.

        args:
            new_values: the array of new values
        """
        self._values = self._values.set(new_values)

    def get_values(self) -> jnp.ndarray:
        """Return the values of these design parameters."""
        return self._values

    def get_values_np(self) -> np.ndarray:
        """Return the values of these design parameters."""
        return np.array(self._values)
