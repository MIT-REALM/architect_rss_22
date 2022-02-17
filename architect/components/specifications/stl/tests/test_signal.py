import jax
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


def test_SampledSignal_stack_grad():
    # Create two signals to test
    t1 = jnp.arange(0.0, 3, 0.3)
    x1 = t1 ** 2
    signal1 = SampledSignal(t1, x1)
    t2 = jnp.arange(0.0, 3, 0.5)
    x2 = t2 ** 0.5

    stack_f = lambda x: SampledSignal.stack(signal1, SampledSignal(t2, x)).x[0, 1]
    stacked_x_grad = jax.grad(stack_f)(x2)

    assert stacked_x_grad.shape == x2.shape
    assert jnp.allclose(stacked_x_grad, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_SampledSignal_stack_vmap():
    # Create two signals to test
    t1 = jnp.arange(0.0, 3, 0.3)
    x1 = t1 ** 2
    signal1 = SampledSignal(t1, x1)
    t2 = jnp.arange(0.0, 3, 0.5)
    x2 = t2 ** 0.5

    stack_f = lambda x: SampledSignal.stack(signal1, SampledSignal(t2, x)).x
    stacked_x_vmap = jax.vmap(stack_f, in_axes=0)(jnp.vstack([x2] * 2))

    assert stacked_x_vmap.shape == (2, t1.size + t2.size - 2, 2)


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


def test_SampledSignal_max1d():
    # Create two signals to test
    t = jnp.arange(0.0, 3.0, 0.3)
    x1 = 0.5 - t
    x2 = 0.0 * t
    signal1 = SampledSignal(t, x1)
    signal2 = SampledSignal(t, x2)

    smoothing = 1e6
    signal = SampledSignal.max1d(signal1, signal2, smoothing)

    assert signal.n == 1
    assert signal.T == t.size + 1  # 1 intersection


def test_SampledSignal_max1d_grad():
    # Create two signals to test
    t = jnp.arange(0.0, 3.0, 0.3)
    x1 = 0.5 - t
    x2 = 0.0 * t
    signal1 = SampledSignal(t, x1)

    smoothing = 1e6
    max_sum_f = lambda x: SampledSignal.max1d(
        signal1, SampledSignal(t, x), smoothing
    ).x.sum()
    max_sum_grad = jax.grad(max_sum_f)(x2)

    assert max_sum_grad.shape == x2.shape
    # With large smoothing constant, the max should not depend on x2 at the first
    # element (where x1 is bigger), but it should depend on x2 at the last element
    # (where x2 is bigger)
    assert jnp.isclose(max_sum_grad[0], 0.0)
    assert jnp.isclose(max_sum_grad[-1], 1.0)


def test_SampledSignal_max1d_vmap():
    # Create two signals to test
    t = jnp.arange(0.0, 3.0, 0.3)
    x1 = 0.5 - t
    x2 = 0.0 * t
    signal1 = SampledSignal(t, x1)

    smoothing = 1e6
    # vmap currently only works with interpolate=False
    max_sum_f = lambda x: SampledSignal.max1d(
        signal1, SampledSignal(t, x), smoothing, interpolate=False
    ).x.sum()
    max_sum_vmap = jax.vmap(max_sum_f, in_axes=0)(jnp.vstack([x2] * 3))

    assert max_sum_vmap.size == 3
