import jax.numpy as jnp

from architect.design import BoundedDesignParameters


class SatDesignParameters(BoundedDesignParameters):
    """BoundedDesignParameters for the satellite rendezvous STL problem"""

    def __init__(self, time_steps):
        """
        Initialize the design parameters for the satellite rendezvous STL problem.
        """
        # Make an array of names
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

        # Make an array of bounds
        bounds = jnp.zeros((n_params, 2))
        bounds = bounds.at[:, 0].set(-100.0)
        bounds = bounds.at[:, 1].set(100.0)

        super(SatDesignParameters, self).__init__(n_params, bounds, names)

        # Initialize the control gains
        gains = jnp.array(
            [
                [50.0, 0.0, 0.0, 50.0, 0.0, 0.0],
                [0.0, 50.0, 0.0, 0.0, 50.0, 0.0],
                [0.0, 0.0, 50.0, 0.0, 0.0, 50.0],
            ]
        )
        self._values = self._values.at[: n_controls * n_states].set(gains.reshape(-1))
