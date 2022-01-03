from typing import Optional, Tuple, Sequence

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import DesignParameters


class MAMDesignParameters(DesignParameters):
    """ExogenousParameters for the multi-agent manipulation task"""

    def __init__(
        self, prng_key: PRNGKeyArray, layer_widths: Tuple[int], n_turtles: int = 2
    ):
        """
        Initialize the design parameters for the multi-agent manipulation problem.
        """
        # Sanity checks
        assert layer_widths[0] == n_turtles * 3 + 3
        assert layer_widths[-1] == n_turtles * 2

        # Figure out how many total parameters we need
        n_layers = len(layer_widths)
        n_weight_params = 0
        n_bias_params = 0
        names = []
        for i in range(1, n_layers):
            input_width = layer_widths[i - 1]
            output_width = layer_widths[i]
            n_weight_params += input_width * output_width
            n_bias_params += output_width

            names += [
                f"layer_{i}_w[{j},{k}]"
                for k in range(input_width)
                for j in range(output_width)
            ]
            names += [f"layer_{i}_b[{j}]" for j in range(output_width)]

        super(MAMDesignParameters, self).__init__(
            n_weight_params + n_bias_params + 2, names
        )

        # Initialize these weights and biases randomly
        self._values = 0.1 * jax.random.normal(prng_key, shape=(self.size,))

        # Set the high level control gains (the first two parameters) to sensible
        # defaults
        self._values = self._values.at[:2].set(jnp.array([10.0, 10.0]))

    @property
    def bounds(self) -> Sequence[Tuple[Optional[float], Optional[float]]]:
        """Returns the bounds on the design parameters as a list. Each element
        of the list should be None (indicates no bound) or a tuple of (lower, upper)
        bounds.

        The first two parameters are bounded above to avoid excessive control
        inputs. The rest of the parameters are unbounded
        """
        return [(None, 10.0)] * 2 + [(None, None)] * (self.size - 2)  # type: ignore
