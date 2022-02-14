"""Defines a notion of discrete-time signals used by STL specifications"""
import jax.numpy as jnp
from jax.nn import logsumexp

import time


class SampledSignal(object):
    """SampledSignal defines a discrete-time continuous-value signal."""

    def __init__(self, time_trace: jnp.ndarray, state_trace: jnp.ndarray):
        """
        Initialize a SampledSignal.

        A SampledSignal has two data members, x and t, which together represent
        timestamped samples of a continuous signal; i.e., x[i] is the value of the
        signal sampled at t[i].

        args:
            time_trace: (T,) array of timestamps for corresponding state samples
            state_trace: (T, n) array of m samples of a signal in R^n
        """
        super(SampledSignal, self).__init__()

        # Save the signals after making sure they match
        if state_trace.shape[0] != time_trace.shape[0]:
            raise ValueError(
                (
                    "state_trace and time_trace must have the same number of samples "
                    f"(state_trace has {state_trace.shape[0]} while time_trace has "
                    f"{time_trace.shape[0]})"
                )
            )

        # Make sure x is multi-dimensional
        self.x = state_trace.reshape(time_trace.shape[0], -1)
        self.t = time_trace

    def stack(*signals: "SampledSignal") -> "SampledSignal":
        """Stack the given signals into a single signal with n = sum_i signals[i].n

        If the given signals do not have the same time indices, they will be linearly
        interpolated at all times in the union of their time indices.

        args:
            signals: a list of SampledSignal objects that will be stacked.
        """
        # Get the union of time indices
        new_t = jnp.array([]).reshape(-1)
        for signal in signals:
            new_t = jnp.union1d(new_t, signal.t)

        # Get the signal values by interpolating at the new time indices
        new_n = sum([signal.n for signal in signals])
        new_T = new_t.shape[0]
        new_x = jnp.zeros((new_T, new_n))
        current_idx = 0
        for signal in signals:
            for i in range(signal.n):
                new_x = new_x.at[:, current_idx].set(
                    jnp.interp(new_t, signal.t, signal.x[:, i])
                )
                current_idx += 1

        return SampledSignal(new_t, new_x)

    def max1d(
        signal_1: "SampledSignal", signal_2: "SampledSignal", smoothing: float
    ) -> "SampledSignal":
        """Take the elementwise smoothed maximum (log-sum-exp) of the two signals.

        Each signal must have n = 1. This function will assume each signal is piecewise
        affine to resample appropriately before comparing the signal. The result is not
        guaranteed to have the same timestamps as either input signal.

        args:
            signal_1: First signal to compare. Must have n = 1
            signal_1: First signal to compare. Must have n = 1
            smoothing: the parameter determining how much to smoothly approximate the
            max of two signals. Uses the log-sum-exp approximation.
        returns:
            a signal with n = 1 containing the smoothed maximum of the two signals
        """
        # Check the inputs
        if signal_1.n > 1 or signal_2.n > 1:
            raise ValueError(
                "Input signals must both be 1-dimensional (got {} and {})".format(
                    signal_1.n, signal_2.n
                )
            )

        # Re-sample the signals to the same timestamps
        merged_s = SampledSignal.stack(signal_1, signal_2)

        # To make sure we don't introduce conservatism, we need to add a timestamped
        # sample at the intersection between these two signals. First, find all
        # intersections by looking at when signal_1.x - signal_2.x changes sign
        diff = merged_s.x[:, 0] - merged_s.x[:, 1]
        # This will contain the indices of zero crossings
        zero_crossings = jnp.where(jnp.diff(jnp.signbit(diff)))[0]

        # At each intersection, we need to add a new timestamped sample for each signal
        new_timestamps = []
        for i in zero_crossings:
            # Get start time and duration of the interval containing the intersection
            intersection_t = merged_s.t[i]
            dt = merged_s.t[i + 1] - intersection_t

            # Get the slope of each signal in this domain.
            x1_start = merged_s.x[i, 0]
            x1_end = merged_s.x[i + 1, 0]
            dx1_dt = (x1_end - x1_start) / dt
            x2_start = merged_s.x[i, 1]
            x2_end = merged_s.x[i + 1, 1]
            dx2_dt = (x2_end - x2_start) / dt

            # Use the slopes to compute the time of the intersection relative to the
            # start time
            dt_intersect = (x2_start - x1_start) / (dx1_dt - dx2_dt)
            t_intersect = intersection_t + dt_intersect
            new_timestamps.append(t_intersect)

        # Add these new samples to both signals by intepolating
        new_t = jnp.array(new_timestamps)
        new_t = jnp.union1d(new_t, merged_s.t)
        new_x = jnp.zeros((new_t.size, 2))
        new_x = new_x.at[:, 0].set(jnp.interp(new_t, merged_s.t, merged_s.x[:, 0]))
        new_x = new_x.at[:, 1].set(jnp.interp(new_t, merged_s.t, merged_s.x[:, 1]))

        # Take the elementwise smoothed maximum
        smoothed_max = 1 / smoothing * logsumexp(smoothing * new_x, axis=1)

        # Return a new signal with this smoothed max
        return SampledSignal(new_t, smoothed_max)

    @property
    def n(self) -> int:
        """Return the dimension of the continuous signal"""
        return self.x.shape[1]

    @property
    def T(self) -> int:
        """Return the number of samples in the signal"""
        return self.x.shape[0]

    @property
    def t_T(self) -> int:
        """Return the timestamp of the last sample"""
        return self.t[-1]

    # Define some magic methods for convenience
    def __len__(self) -> int:
        return self.T

    def __getitem__(self, position) -> "SampledSignal":
        return SampledSignal(self.t[position].reshape(-1), self.x[position])

    # Left and right multiplication apply to the sampled values
    def __mul__(self, factor: jnp.ndarray) -> "SampledSignal":
        return SampledSignal(self.t, self.x * factor)

    # Define negation
    def __neg__(self) -> "SampledSignal":
        return -1 * self

    __rmul__ = __mul__

    def subsignal_time(
        self, t_start: float, t_end: float = float("inf")
    ) -> "SampledSignal":
        """Return the subsignal containing samples with timestamps in the given interval.

        Note: this function identifies subsignals using an interval of real-valued
        timestamps. If you wish instead to identify subsignals by integer index of
        discrete samples, use the Python slicing syntax `signal[i:j]`

        This function is not compatible with jax.jit

        args:
            t_start: starting time of the subsignal interval (inclusive)
            t_end: Optional; if provided, denotes the ending time of the subsignal
                interval (exclusive). If not specified, defaults to include the rest of
                the signal.
        returns:
            a new SampledSignal containing the subsignal
        """
        if t_end <= t_start:
            raise ValueError("t_end must be strictly greater than t_start")

        interval_mask = jnp.logical_and(self.t >= t_start, self.t < t_end)
        return SampledSignal(self.t[interval_mask], self.x[interval_mask])
