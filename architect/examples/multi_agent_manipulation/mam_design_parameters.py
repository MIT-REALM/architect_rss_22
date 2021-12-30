from typing import List

import jax
from jax._src.prng import PRNGKeyArray

from architect.design import DesignParameters


class MAMDesignParameters(DesignParameters):
    """ExogenousParameters for the multi-agent manipulation task"""

    def __init__(
        self, prng_key: PRNGKeyArray, layer_widths: List[int], n_turtles: int = 2
    ):
        """
        Initialize the design parameters for the multi-agent manipulation problem.
        """
        # Sanity checks
        assert layer_widths[0] == n_turtles * 6 + 3
        assert layer_widths[-1] == n_turtles * 3

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
            n_weight_params + n_bias_params, names
        )

        # Initialize these weights and biases randomly
        self._values = 0.1 * jax.random.normal(prng_key, shape=(self.size,))
