import time

import jax.numpy as jnp
import matplotlib.pyplot as plt  # noqa


from architect.components.specifications.stl.formula import (
    STLPredicate,
    STLAnd,
)


def make_test_predicate():
    # Test if the signal has absolute value less than 0.1
    lower_bound = 0.1
    mu = lambda x_t: -jnp.abs(x_t)
    return STLPredicate(mu, -lower_bound)


def make_test_signal(dt=0.01):
    t = jnp.arange(0.0, 10.0, dt)
    x = jnp.sin(4 * t) * jnp.exp(-t / 2)
    return jnp.vstack((t, x))


def main():
    # Construct a test signal and predicate
    signal = make_test_signal(0.001)
    p = make_test_predicate()
    p = STLAnd(p, p, interpolate=False)

    # Compute robustness
    p(signal)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(end - start)
