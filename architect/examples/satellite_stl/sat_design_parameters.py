from typing import Sequence, Tuple, Optional

import jax.numpy as jnp

from architect.design import DesignParameters


class SatDesignParameters(DesignParameters):
    """DesignParameters for the satellite rendezvous STL problem"""

    def __init__(self, time_steps):
        """
        Initialize the design parameters for the satellite rendezvous STL problem.
        """
        n_states = 6
        n_controls = 3
        dims = {"x": n_states, "u": n_controls}
        names = [f"k_{i}_{j}" for i in range(n_controls) for j in range(n_states)]
        names += [
            f"{name}_ref_{i}({t})"
            for t in range(time_steps)
            for name in ["u", "x"]
            for i in range(dims[name])
        ]
        n_params = n_controls * n_states + time_steps * (n_controls + n_states)
        super(SatDesignParameters, self).__init__(n_params, names)

        # Initialize the control gains
        gains = jnp.array(
            [
                [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0, 0.0, 100.0, 0.0],
                [0.0, 0.0, 100.0, 0.0, 0.0, 100.0],
            ]
        )
        self._values = self._values.at[:n_controls * n_states].set(gains.reshape(-1))

    @property
    def bounds(self) -> Sequence[Tuple[Optional[float], Optional[float]]]:
        """Returns the bounds on the design parameters as a list. Each element
        of the list should be None (indicates no bound) or a tuple of (lower, upper)
        bounds.

        Default behavior (unless overridden by a subclass) is to not bound any variables
        """
        return [(None, None)] * self.size
