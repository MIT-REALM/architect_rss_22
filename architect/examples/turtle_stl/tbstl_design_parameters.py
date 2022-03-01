import jax.numpy as jnp

from architect.design import BoundedDesignParameters


class TBSTLDesignParameters(BoundedDesignParameters):
    """BoundedDesignParameters for the turtlebot rendezvous STL problem"""

    def __init__(self, time_steps):
        """
        Initialize the design parameters for the turtlebot rendezvous STL problem.
        """
        # Make an array of names
        n_controls = 2
        names = [
            f"u_ref_{i}({t})" for t in range(time_steps) for i in range(n_controls)
        ]
        n_params = time_steps * n_controls

        # Make an array of bounds
        bounds = jnp.zeros((time_steps, n_controls, 2))
        bounds = bounds.at[:, 0, 0].set(0.1)  # v lower bound
        bounds = bounds.at[:, 0, 1].set(0.5)  # v upper bound
        bounds = bounds.at[:, 1, 0].set(-1.0)  # w lower bound
        bounds = bounds.at[:, 1, 1].set(1.0)  # w upper bound
        bounds = bounds.reshape(-1, 2)

        super(TBSTLDesignParameters, self).__init__(n_params, bounds, names)
