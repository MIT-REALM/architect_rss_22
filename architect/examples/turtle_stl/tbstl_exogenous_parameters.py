import jax.numpy as jnp

from architect.design import BoundedExogenousParameters


class TBSTLExogenousParameters(BoundedExogenousParameters):
    """BoundedExogenousParameters for the multi-agent manipulation task"""

    def __init__(self):
        """
        Initialize the exogenous parameters for the satellite STL problem.
        """
        n_vars = 3
        names = ["px0", "py0", "theta0"]
        bounds = jnp.array(
            [
                [-1.25, -1.15],  # px
                [-1.55, -1.45],  # py
                [jnp.pi / 2 - 0.01, jnp.pi / 2 + 0.01],  # theta
            ]
        )

        super(TBSTLExogenousParameters, self).__init__(n_vars, bounds, names)
