import jax.numpy as jnp

from architect.components.specifications.stl.signal import SampledSignal


def test_SampledSignal_init():
    # Create a signal to test
    t = jnp.arange(0.0, 10.0, 0.1)
    x = t ** 2
    signal = SampledSignal(t, x)

    assert jnp.allclose(signal.x, x.reshape(-1, 1))
    assert jnp.allclose(signal.t, t)

    assert signal.n == 1
    assert signal.T == t.shape[0]
    assert signal.t_T == t[-1]

    # Try again with a 3D signal
    x = jnp.vstack((x, 2 * x, 3 * x)).T
    signal = SampledSignal(t, x)

    assert signal.n == 3
    assert signal.T == t.shape[0]


def test_SampledSignal_stack():
    # Create two signals to test
    t1 = jnp.arange(0.0, 3, 0.3)
    x1 = t1 ** 2
    signal1 = SampledSignal(t1, x1)
    t2 = jnp.arange(0.0, 3, 0.5)
    x2 = t2 ** 0.5
    signal2 = SampledSignal(t2, x2)

    signal = SampledSignal.stack(signal1, signal2)

    assert signal.n == 2
    assert signal.T == t1.size + t2.size - 2  # overlap at 0.0 and 1.5


def test_SampledSignal_getitem():
    # Create a signal to test
    t = jnp.arange(0.0, 10.0, 0.1)
    x = t ** 2
    signal = SampledSignal(t, x)

    # Test getting a single element
    i = 2
    subsignal = signal[i]
    assert jnp.allclose(subsignal.t, t[i].reshape(-1))
    assert jnp.allclose(subsignal.x, x[i].reshape(-1, 1))

    # Test getting a slice
    T = 5
    subsignal = signal[i : i + T]
    assert subsignal.T == T
    assert subsignal.x.shape == (T, 1)
    assert jnp.allclose(subsignal.t, t[i : i + T].reshape(-1))
    assert jnp.allclose(subsignal.x, x[i : i + T].reshape(-1, 1))


def test_SampledSignal_mul():
    # Create a signal to test
    t = jnp.arange(0.0, 10.0, 0.1)
    x = t ** 2
    signal = SampledSignal(t, x)

    signal2 = signal * 2
    assert jnp.allclose(signal2.x, 2 * signal.x)
    signal2 = 2 * signal
    assert jnp.allclose(signal2.x, 2 * signal.x)

    signal2 = signal * t.reshape(-1, 1)
    assert jnp.allclose(signal2.x.squeeze(), t ** 3)


def test_SampledSignal_subsignal_time():
    # Create a signal to test
    t = jnp.arange(0.0, 10.0, 0.1)
    x = t ** 2
    signal = SampledSignal(t, x)

    # Test default behavior
    t_start = 5.0
    subsignal = signal.subsignal_time(t_start)
    assert jnp.min(subsignal.t) == t_start
    assert subsignal.x.shape[0] == subsignal.T

    # Test behavior with set endpoint
    t_end = 7.0
    subsignal = signal.subsignal_time(t_start, t_end)
    assert jnp.min(subsignal.t) == t_start
    assert jnp.max(subsignal.t) == t_end - 0.1
    assert subsignal.x.shape[0] == subsignal.T
