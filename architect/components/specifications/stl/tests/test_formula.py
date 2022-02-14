import jax.numpy as jnp
import matplotlib.pyplot as plt  # noqa


from architect.components.specifications.stl.signal import SampledSignal
from architect.components.specifications.stl.formula import (
    STLPredicate,
    STLNegation,
    STLAnd,
    STLOr,
    STLImplies,
    STLUntimedEventually,
    STLTimedEventually,
    STLUntimedUntil,
    STLUntimedAlways,
    STLTimedAlways,
)


def make_test_predicate():
    # Test if the signal has absolute value less than 0.1
    lower_bound = 0.1
    mu = lambda x_t: -jnp.abs(x_t)
    return STLPredicate(mu, -lower_bound)


def make_test_signal(dt=0.01):
    t = jnp.arange(0.0, 10.0, dt)
    x = jnp.sin(4 * t) * jnp.exp(-t / 2)
    return SampledSignal(t, x)


def test_STLPredicate():
    # Construct a test signal
    signal = make_test_signal()

    # Make a predicate to test if the signal is smaller than 0.1
    p = make_test_predicate()

    # Compute robustness
    r = p(signal)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics
    assert ((r.x > 0) == (jnp.abs(signal.x) < 0.1).reshape(-1, 1)).all()

    # Also test on a multi-dimensional signal
    x2 = jnp.vstack((signal.x.squeeze(), signal.x.squeeze(), signal.x.squeeze())).T
    signal = SampledSignal(signal.t, x2)
    mu = lambda x_t: -jnp.abs(x_t[0])
    p = STLPredicate(mu, -0.1)

    # Compute robustness
    r = p(signal)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics
    assert ((r.x > 0) == (jnp.abs(signal.x[:, 0]) < 0.1).reshape(-1, 1)).all()


def test_STLNegation():
    # Construct a test signal
    signal = make_test_signal()

    # Make a predicate to test if the signal is smaller than 0.1
    p = make_test_predicate()

    # Make a negation
    not_p = STLNegation(p)

    # Get robustness
    r = not_p(signal)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics
    assert ((r.x > 0) != (jnp.abs(signal.x) < 0.1).reshape(-1, 1)).all()


def test_STLAnd():
    # Construct a test signal
    signal = make_test_signal()

    # Make a predicate to test if the signal is smaller than 0.1
    p1 = make_test_predicate()

    # Make a second test predicate to test if the signal is positive
    mu = lambda x_t: x_t
    p2 = STLPredicate(mu, 0.0)

    # And them together
    p_and = STLAnd([p1, p2])

    # Compute robustness
    r = p_and(signal, smoothing=1e6)

    # Check shapes
    assert r.x.shape == (r.T, 1)

    # Check semantics
    compare_mask = jnp.searchsorted(r.t, signal.t)
    satisfied = jnp.logical_and(jnp.abs(signal.x) < 0.1, signal.x > 0.0)

    assert ((r.x[compare_mask] > 0) == satisfied).all()

    # Test with coarser time samples
    t = jnp.arange(0.0, 1.0, 0.2)
    x1 = 1.0 - 2.0 * t
    x2 = -0.1 + t / 3.0
    x = jnp.vstack((x1, x2)).T
    signal = SampledSignal(t, x)

    # Make a formula for when both signals are positive
    mu_1 = lambda x_t: x_t[0]
    mu_2 = lambda x_t: x_t[1]
    p1 = STLPredicate(mu_1, 0.0)
    p2 = STLPredicate(mu_2, 0.0)
    p_and = STLAnd([p1, p2])

    # Get robustness
    r = p_and(signal, smoothing=1e6)

    # The robustness trace should have added a point for the intersection between these
    # lines
    assert r.T == signal.T + 1


def test_STLOr():
    # Construct a test signal
    signal = make_test_signal()

    # Make a predicate to test if the signal is smaller than 0.1
    p1 = make_test_predicate()

    # Make a second test predicate to test if the signal is positive
    mu = lambda x_t: x_t
    p2 = STLPredicate(mu, 0.0)

    # Or them together
    p_or = STLOr([p1, p2])

    # Compute robustness
    r = p_or(signal)

    # Check shapes
    assert r.x.shape == (r.T, 1)

    # Check semantics
    compare_mask = jnp.searchsorted(r.t, signal.t)
    satisfied = jnp.logical_or(jnp.abs(signal.x) < 0.1, signal.x > 0.0)
    assert ((r.x[compare_mask] > 0) == satisfied).all()


def test_STLImplies():
    # Construct a test signal so that the two dimensions always
    # have the same sign
    t = jnp.arange(0.0, 0.1, 0.01)
    x1 = jnp.sin(4 * t)
    x2 = 2 * jnp.sin(2 * 4 * t)
    x = jnp.vstack((x1, x2)).T
    signal = SampledSignal(t, x)

    # Make two predicates to test if each dimension of the signal is positive
    mu_1 = lambda x_t: x_t[0]
    p1 = STLPredicate(mu_1, 0.0)
    mu_2 = lambda x_t: x_t[1]
    p2 = STLPredicate(mu_2, 0.0)

    # Make a predicate saying that the second dimension is positive whenever the first
    # dimension is positive.
    p = STLImplies(p1, p2)

    # Compute robustness
    r = p(signal)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics. Make use of the fact that (A -> B) <-> ((not A) or B)
    compare_mask = jnp.searchsorted(r.t, signal.t)
    satisfied = jnp.logical_or(
        jnp.logical_not(signal.x[:, 0] > 0), signal.x[:, 1] > 0.0
    )
    assert ((r.x[compare_mask] > 0).squeeze() == satisfied).all()


def test_STLUntimedEventually():
    # Construct a test signal
    signal = make_test_signal()

    # Make a predicate to test if the absolute value of the signal is large
    mu = lambda x_t: jnp.abs(x_t)
    p_large = STLPredicate(mu, 0.1)

    # Make a predicate saying that the signal eventually grows large (between zero and
    # infinite time from now)
    p_eventually_large = STLUntimedEventually(p_large)

    # Compute robustness
    r = p_eventually_large(signal, smoothing=1e6)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics. This signal falls below 0.1 for the last time at t \approx 4.42
    satisfied_mask = r.t <= 4.42
    unsatisfied_mask = r.t >= 4.43
    assert (r.x[satisfied_mask] > 0).all()
    assert (r.x[unsatisfied_mask] < 0).all()

    # # Plot
    # plt.plot(signal.t, signal.x)
    # plt.plot(r.t, r.x)
    # plt.plot(r.t, r.x > 0.0)
    # plt.plot(signal.t, signal.t * 0, "k:")
    # plt.plot(r.t, r.t * 0 + 0.1, "k:")
    # plt.plot(r.t, r.t * 0 - 0.1, "k:")
    # plt.show()


def test_STLTimedEventually():
    # Construct a test signal
    t = jnp.arange(0, 5.0, 0.1)
    x = -2.5 + t
    signal = SampledSignal(t, x)

    # Make a predicate to test if the absolute value of the signal is small
    mu = lambda x_t: -jnp.abs(x_t)
    p_small = STLPredicate(mu, -0.1)

    # Make a predicate saying that the signal eventually grows small (between 1 and 1.1
    # seconds from now)
    p_eventually_small = STLTimedEventually(p_small, 1.0, 1.1)

    # Compute robustness
    r = p_eventually_small(signal, smoothing=1e6)

    # # Plot
    # plt.plot(signal.t, signal.x)
    # plt.plot(r.t, r.x, marker="o")
    # plt.plot(r.t, r.x > 0.0, marker="o")
    # plt.plot(signal.t, signal.t * 0, "k:")
    # plt.plot(r.t, r.t * 0 + 0.1, "k:")
    # plt.plot(r.t, r.t * 0 - 0.1, "k:")
    # plt.show()

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics. This should be satisied between 1.3 and 1.6 (since the signal is
    # small between 2.4 and 2.6)
    satisfied_mask = jnp.logical_and(r.t > 1.3, r.t < 1.6)
    unsatisfied_mask = jnp.logical_or(r.t < 1.3, r.t > 1.6)
    assert (r.x[satisfied_mask] > 0).all()
    assert (r.x[unsatisfied_mask] < 0).all()


def test_STLUntimedUntil():
    # Construct test signals
    signal0 = make_test_signal(dt=0.5)
    x1 = 1.1 - signal0.t / 5
    x2 = -1.0 + signal0.t / 5
    x = jnp.vstack((x1, x2)).T
    signal = SampledSignal(signal0.t, x)

    # Make a predicate to test if a signal is positive
    mu1 = lambda x_t: x_t[0]
    p_pos1 = STLPredicate(mu1, 0.0)
    mu2 = lambda x_t: x_t[1]
    p_pos2 = STLPredicate(mu2, 0.0)

    # Make a formula for signal1 positive until signal2 is positive
    p_until = STLUntimedUntil(p_pos1, p_pos2)

    # Compute robustness
    r = p_until(signal, smoothing=1e6)

    # Check shapes
    assert r.T == signal.T + 1  # 1 intersection point got added
    assert r.x.shape == (r.T, 1)

    # Check semantics. Satisfied until t = 5.5
    satisfied_mask = r.t <= 5.49
    unsatisfied_mask = r.t > 5.5
    assert (r.x[satisfied_mask] > 0).all()
    assert (r.x[unsatisfied_mask] < 0).all()


def test_STLUntimedAlways():
    # Construct a test signal
    signal = make_test_signal()

    # Make a predicate to test if the absolute value of the signal is small
    mu = lambda x_t: -jnp.abs(x_t)
    p_small = STLPredicate(mu, -0.1)

    # Make a predicate saying that the signal is always small (between zero and
    # infinite time from now)
    p_always_small = STLUntimedAlways(p_small)

    # Compute robustness
    r = p_always_small(signal, smoothing=1e6)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics. This signal falls below 0.1 for the last time at t \approx 4.42
    satisfied_mask = r.t >= 4.43
    unsatisfied_mask = r.t <= 4.42
    assert (r.x[satisfied_mask] > 0).all()
    assert (r.x[unsatisfied_mask] < 0).all()

    # # Plot
    # plt.plot(signal.t, signal.x)
    # plt.plot(r.t, r.x)
    # plt.plot(r.t, r.x > 0.0)
    # plt.plot(signal.t, signal.t * 0, "k:")
    # plt.plot(r.t, r.t * 0 + 0.1, "k:")
    # plt.plot(r.t, r.t * 0 - 0.1, "k:")
    # plt.show()


def test_STLTimedAlways():
    # Construct a test signal
    # Construct a test signal
    t = jnp.arange(0, 5.0, 0.05)
    x = -2.5 + t
    signal = SampledSignal(t, x)

    # Make a predicate to test if the absolute value of the signal is small
    mu = lambda x_t: -jnp.abs(x_t)
    p_small = STLPredicate(mu, -0.2)

    # Make a predicate saying that the signal stays small (between 1 and 1.1
    # seconds from now)
    p_always_small = STLTimedAlways(p_small, 1.0, 1.1)

    # Compute robustness
    r = p_always_small(signal, smoothing=1e6)

    # Check shapes
    assert r.T == signal.T
    assert r.x.shape == (r.T, 1)

    # Check semantics. This signal satisfies from 1.3 to 1.6 s
    satisfied_mask = jnp.logical_and(r.t > 1.31, r.t < 1.59)
    unsatisfied_mask = jnp.logical_or(r.t < 1.3, r.t > 1.6)
    assert (r.x[satisfied_mask] > 0).all()
    assert (r.x[unsatisfied_mask] < 0).all()

    # # Plot
    # plt.plot(signal.t, signal.x)
    # plt.plot(r.t, r.x)
    # plt.plot(r.t, r.x > 0.0)
    # plt.plot(signal.t, signal.t * 0, "k:")
    # plt.plot(r.t, r.t * 0 + 0.2, "k:")
    # plt.plot(r.t, r.t * 0 - 0.2, "k:")
    # plt.show()
