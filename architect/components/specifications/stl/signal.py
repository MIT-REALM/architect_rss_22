"""Defines a notion of discrete-time signals used by STL specifications"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.nn import logsumexp


@jax.jit
def stack(signal_1: jnp.ndarray, signal_2: jnp.ndarray) -> jnp.ndarray:
    """Stack the given signals into a single signal

    If the given signals do not have the same time indices, they will be linearly
    interpolated at all times in the union of their time indices.

    Output will have the same number of samples as the first input signal.

    args:
        signal_1, signal_2: jnp arrays where the first row contains the timestamps
            and subsequent rows contain the sampled values. Samples are grouped by
            column
    """
    # Interpolate the second signal to match the first signal
    new_s = jnp.zeros((signal_2.shape[0], signal_1.shape[1]))
    new_s = new_s.at[0].set(signal_1[0])
    for i in range(1, signal_2.shape[0]):
        new_s = new_s.at[i].set(jnp.interp(signal_1[0], signal_2[0], signal_2[i]))

    stacked_signal = jnp.vstack((signal_1, new_s[1:]))

    return stacked_signal


@partial(jax.jit, static_argnames=["interpolate"])
def max1d(
    signal_1: jnp.ndarray,
    signal_2: jnp.ndarray,
    smoothing: float,
    interpolate: bool = True,
) -> jnp.ndarray:
    """Take the elementwise smoothed maximum (log-sum-exp) of the two signals.

    Each signal must have one non-time row. This function will assume each signal is
    piecewise affine to resample appropriately before comparing the signal. The result
    is not guaranteed to have the same timestamps as either input signal; it may have
    new or duplicate timestamps.

    args:
        signal_1, signal_2: jnp arrays to compare where the first row contains the
            timestamps and subsequent rows contain the sampled values. Samples are
            grouped by column.
        smoothing: the parameter determining how much to smoothly approximate the
            max of two signals. Uses the log-sum-exp approximation.
        interpolate: if False, only compare signals at the given timestamps, don't
            look for intersections between points
    returns:
        a signal with shape[0] = 2 containing the smoothed maximum of the two signals
    """
    # Check the inputs
    if signal_1.shape[0] != 2 or signal_2.shape[0] != 2:
        raise ValueError(
            "Input signals must both have 2 rows (got {} and {})".format(
                signal_1.shape[0], signal_2.shape[0]
            )
        )

    # Re-sample the signals to the same timestamps
    merged_s = stack(signal_1, signal_2)

    # If we don't need to interpolate, this is very simple
    if not interpolate:
        smoothed_max = 1 / smoothing * logsumexp(smoothing * merged_s[1:], axis=0)
        return jnp.vstack((merged_s[0].reshape(1, -1), smoothed_max))

    # If we need to interpolate, it gets more complicated

    # To make sure we don't introduce conservatism, we need to add a timestamped
    # sample at the intersection between these two signals. First, find all
    # intersections by looking at when signal_1[1] - signal_2[1] changes sign
    diff = merged_s[1] - merged_s[2]
    # This will contain the indices of zero crossings
    zero_crossings = jnp.nonzero(
        jnp.diff(jnp.signbit(diff)), size=diff.size, fill_value=-1
    )[0]

    def t_intersect(i):
        """Return the time of intersection in the interval starting at index i"""
        # Get start time and length of the interval containing the intersection
        intersection_t = merged_s[0, i]
        dt = merged_s[0, i + 1] - intersection_t

        # Get the slope of each signal in this domain.
        x1_start = merged_s[1, i]
        x1_end = merged_s[1, i + 1]
        dx1_dt = (x1_end - x1_start) / (dt + 1e-3)
        x2_start = merged_s[2, i]
        x2_end = merged_s[2, i + 1]
        dx2_dt = (x2_end - x2_start) / (dt + 1e-3)

        # Use the slopes to compute the time of the intersection relative to the
        # start time
        dt_intersect = (x2_start - x1_start) / (dx1_dt - dx2_dt + 1e-3)
        dt_intersect = jnp.nan_to_num(dt_intersect)
        t_intersect = intersection_t + dt_intersect
        return t_intersect

    # At each intersection, we need to add a new sample for each signal
    added_t = jax.vmap(t_intersect)(zero_crossings)

    # Clamp to make sure we only add intersections within the domain of the signal
    added_t = jnp.clip(added_t, merged_s[0, 0], merged_s[0, -1])

    # Add these new samples to both signals by intepolating
    new_t = jnp.union1d(merged_s[0], added_t, size=added_t.size + merged_s[0].size)
    new_t = jnp.sort(new_t)
    new_x = jnp.zeros((2, new_t.size))
    new_x = new_x.at[0, :].set(jnp.interp(new_t, merged_s[0], merged_s[1]))
    new_x = new_x.at[1, :].set(jnp.interp(new_t, merged_s[0], merged_s[2]))
    merged_s = jnp.vstack((new_t.reshape(1, -1), new_x))

    # Take the elementwise smoothed maximum
    smoothed_max = 1 / smoothing * logsumexp(smoothing * merged_s[1:], axis=0)

    # Return a new signal with this smoothed max
    return jnp.vstack((merged_s[0].reshape(1, -1), smoothed_max))
