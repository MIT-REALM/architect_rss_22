import jax.numpy as jnp

from architect.design import BoundedExogenousParameters


class SatExogenousParameters(BoundedExogenousParameters):
    """BoundedExogenousParameters for the multi-agent manipulation task"""

    def __init__(self):
        """
        Initialize the exogenous parameters for the satellite STL problem.
        """
        n_vars = 6
        names = ["px0", "py0", "pz0", "vx0", "vy0", "vz0"]
        bounds = jnp.array(
            [
                [10.0, 15.0],  # px
                [10.0, 15.0],  # py
                [-5.0, 5.0],  # pz
                [-1.0, 1.0],  # vx
                [-1.0, 1.0],  # vy
                [-1.0, 1.0],  # vz
            ]
        )

        super(SatExogenousParameters, self).__init__(n_vars, bounds, names)
