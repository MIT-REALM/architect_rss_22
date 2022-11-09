import jax.numpy as jnp
import numpy as np

from architect.design.problem import DesignParameters


def test_DesignParameters_init():
    """Test initialization of DesignParameters object"""
    # Test with a range of sizes
    sizes = range(1, 10)
    for size in sizes:
        dp = DesignParameters(size)

        # Initialization should be successfull
        assert dp is not None

        # We should have the right initial values
        assert jnp.allclose(dp.get_values(), jnp.zeros(size))


def test_DesignParameters_get_set_values():
    """Test getting and setting DesignParameter values"""
    # Create test parameters
    size = 5
    dp = DesignParameters(size)

    # Test getting the values: should get a JAX array of the right shape
    values = dp.get_values()
    assert isinstance(values, jnp.ndarray)
    assert values.shape == (size,)

    # Also test getting values as numpy array
    values_np = dp.get_values_np()
    assert isinstance(values_np, np.ndarray)
    assert values_np.shape == (size,)

    # These two values should match
    assert jnp.allclose(values, values_np)

    # Test setting the values
    new_values = jnp.array([i + 0.1 for i in range(size)])
    dp.set_values(new_values)
    assert jnp.allclose(dp.get_values(), new_values)
