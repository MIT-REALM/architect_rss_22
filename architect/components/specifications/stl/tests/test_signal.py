import jax
import jax.numpy as jnp

from architect.components.specifications.stl.signal import stack, max1d


def test_stack():
    # Create two signals to test
    t1 = jnp.arange(0.0, 3, 0.3)
    x1 = t1 ** 2
    signal1 = jnp.vstack((t1, x1))
    t2 = jnp.arange(0.0, 3, 0.5)
    x2 = t2 ** 0.5
    signal2 = jnp.vstack((t2, x2))

    signal = stack(signal1, signal2)

    assert signal.shape[0] == 2 + 1  # 1 time row and 2 signal rows
    assert signal.shape[1] == t1.size


def test_stack_grad():
    # Create two signals to test
    t1 = jnp.arange(0.0, 3, 0.3)
    x1 = t1
    signal1 = jnp.vstack((t1, x1))
    t2 = jnp.arange(0.0, 3, 0.5)
    x2 = 2 * t2

    stack_f = lambda x: stack(signal1, jnp.vstack((t2, x)))[2, 0]
    stacked_x_grad = jax.grad(stack_f)(x2)

    assert stacked_x_grad.shape == x2.shape
    assert jnp.allclose(stacked_x_grad, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_stack_vmap():
    # Create two signals to test
    t1 = jnp.arange(0.0, 3, 0.3)
    x1 = t1 ** 2
    signal1 = jnp.vstack((t1, x1))
    t2 = jnp.arange(0.0, 3, 0.5)
    x2 = t2 ** 0.5

    stack_f = lambda x: stack(signal1, jnp.vstack((t2, x)))
    stacked_x_vmap = jax.vmap(stack_f, in_axes=0)(jnp.vstack([x2] * 2))

    assert stacked_x_vmap.shape == (2, 3, t1.size)


def test_max1d():
    # Create two signals to test
    t = jnp.arange(0.0, 3.0, 0.3)
    x1 = 0.5 - t
    x2 = 0.0 * t
    signal1 = jnp.vstack((t, x1))
    signal2 = jnp.vstack((t, x2))

    smoothing = 1e6
    signal = max1d(signal1, signal2, smoothing)

    assert signal.shape[0] == 2
    assert signal.shape[1] == 2 * t.size  # max1d doubles the length

    signal2_max_mask = signal[0] >= 0.5
    signal1_max_mask = signal[0] < 0.5
    assert jnp.allclose(signal[1, signal2_max_mask], 0.0, atol=1e-5)
    assert jnp.allclose(signal[1, signal1_max_mask], 0.5 - signal[0, signal1_max_mask])


def test_max1d_grad():
    # Create two signals to test
    t = jnp.arange(0.0, 3.0, 0.3)
    x1 = 0.5 - t
    x2 = 0.0 * t
    signal1 = jnp.vstack((t, x1))

    smoothing = 1e6
    max_sum_f = lambda x: max1d(
        signal1, jnp.vstack((t, x)), smoothing, interpolate=False
    )[1].sum()
    max_sum_grad_f = jax.grad(max_sum_f)
    max_sum_grad = max_sum_grad_f(x2)

    assert max_sum_grad.shape == x2.shape
    # With large smoothing constant, the max should not depend on x2 at the first
    # element (where x1 is bigger), but it should depend on x2 at the last element
    # (where x2 is bigger)
    assert jnp.isclose(max_sum_grad[0], 0.0)
    assert jnp.isclose(max_sum_grad[-1], 1.0)


# def test_max1d_vmap():
#     # Create two signals to test
#     t = jnp.arange(0.0, 3.0, 0.3)
#     x1 = 0.5 - t
#     x2 = 0.0 * t
#     signal1 = jnp.vstack((t, x1))

#     smoothing = 1e6
#     max_sum_f = lambda x: max1d(signal1, jnp.vstack((t, x)), smoothing)[1].sum()
#     max_sum_vmap = jax.vmap(max_sum_f, in_axes=0)(jnp.vstack([x2] * 3))

#     assert max_sum_vmap.size == 3
