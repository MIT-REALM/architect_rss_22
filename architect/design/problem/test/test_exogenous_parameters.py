import jax

from architect.design.problem import ExogenousParameters


def test_ExogenousParameters_init():
    """Test initialization of ExogenousParameters object"""
    # Set a PRNG key to use
    key = jax.random.PRNGKey(0)

    # Test with a range of sizes
    sizes = range(1, 10)
    for size in sizes:
        ep = ExogenousParameters(size)

        # Initialization should be successfull
        assert ep is not None

        # If we sample, we should get the right size
        sample = ep.sample(key)
        assert sample.shape == (size,)


def test_ExogenousParameters_sample():
    """Test sampling ExogenousParameter values"""
    # Set a PRNG key to use
    key = jax.random.PRNGKey(0)

    # Create test parameters
    size = 5
    ep = ExogenousParameters(size)

    # Test sampling one value. It should have the right size
    sample = ep.sample(key)
    assert sample.shape == (size,)

    # Test sampling multiple values
    batch = 10
    sample = ep.sample(key, batch_size=batch)
    assert sample.shape == (batch, size)
