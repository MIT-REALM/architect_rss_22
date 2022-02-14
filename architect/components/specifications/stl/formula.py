"""Defines a tree structure for constructing and evaluating STL formulae"""
from abc import ABC, abstractmethod
from typing import Callable, List

import jax
from jax.nn import logsumexp
import jax.numpy as jnp

from .signal import SampledSignal


class STLFormula(ABC):
    """STLFormula defines a formula in bounded-time signal temporal logic (STL) using a
    tree data structure.

    STLFormula is an abstract base class that defines the interface all STL formulae
    should implement.

    NOTE: Assumes that signals are piecewise affine for the purposes of interpolating
    """

    def __init__(self):
        super(STLFormula, self).__init__()

    @abstractmethod
    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> jnp.ndarray:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            single-element array of the robustness value for this formula and the given
            signal.
        """
        raise NotImplementedError()


class STLPredicate(STLFormula):
    """Represents an STL predicate (function from signal to reals)"""

    def __init__(
        self, predicate: Callable[[jnp.ndarray], jnp.ndarray], lower_bound: float
    ):
        """Initialize an STL formula for a predicate.

        args:
            predicate: a callable that takes a single sample and returns a scalar value
            lower_bound: the predicate is satisfied if its value is greater than this
                bound.
        """
        super(STLPredicate, self).__init__()

        self.predicate = predicate
        self.lower_bound = lower_bound

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A predicate has robustness value equal to the scalar result of its function
        minus the lower bound. The complexity of this call is linear in the length
        of the signal.

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Apply the predicate to each sampled value
        mu_x = jax.vmap(self.predicate)(s.x)

        # Compute the robustness over all subsignals
        robustness = mu_x - self.lower_bound

        return SampledSignal(s.t, robustness)


class STLNegation(STLFormula):
    """Represents an STL negation"""

    def __init__(self, child: STLFormula):
        """Initialize an STL formula for a negation.

        args:
            child: the STLFormula to negate
        """
        super(STLNegation, self).__init__()

        self.child = child

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A negation has robustness value equal to the negative of its child's robustness
        value

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        return -1 * self.child(s, smoothing)


class STLAnd(STLFormula):
    """Represents an STL conjunction"""

    def __init__(self, children: List[STLFormula]):
        """Initialize an STL formula for a conjunction

        args:
            children: the list of formulae to and together. Must have exactly two
                elements
        """
        super(STLAnd, self).__init__()

        if len(children) != 2:
            raise ValueError("STLAnd requires 2 children, got {}".format(len(children)))

        self.children = children

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A conjunction has robustness value equal to the minimum of its children's
        robustness values

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Get children's robustness values
        child1_r = self.children[0](s)
        child2_r = self.children[1](s)

        # Use the trick where min(x, y) = -max(-x, -y)
        r = -SampledSignal.max1d(-child1_r, -child2_r, smoothing)

        return r


class STLOr(STLFormula):
    """Represents an STL disjunction"""

    def __init__(self, children: List[STLFormula]):
        """Initialize an STL formula for a disjunction

        args:
            children: the list of formulae to and together. Must have exactly 2 children
        """
        super(STLOr, self).__init__()

        if len(children) != 2:
            raise ValueError("STLOr requires 2 children, got {}".format(len(children)))

        self.children = children

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A disjunction has robustness value equal to the maximum of its children's
        robustness values

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Get children's robustness values
        child1_r = self.children[0](s)
        child2_r = self.children[1](s)

        # Compute the maximum
        r = SampledSignal.max1d(child1_r, child2_r, smoothing)

        return r


class STLImplies(STLFormula):
    """Represents an STL implication"""

    def __init__(self, premise: STLFormula, conclusion: STLFormula):
        """Initialize an STL formula for a implication

        args:
            premise: an STL formula
            conclusion: an STL formula
        """
        super(STLImplies, self).__init__()

        self.premise = premise
        self.conclusion = conclusion

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A implication has robustness value equal to the maximum of the negative of
        the premise's robustness and the conclusions' robustness. This relates to
        the fact that A implies B is equivalent to (not A) or B.

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Get robustness of premise and conclusion
        r_premise = self.premise(s, smoothing)
        r_conclusion = self.conclusion(s, smoothing)

        # Compute the smoothed maximum
        r = SampledSignal.max1d(-r_premise, r_conclusion, smoothing)

        return r


@jax.jit
def accumulate_max(max_so_far, next_element, smoothing):
    comparands = jnp.concatenate((max_so_far.reshape(-1), next_element))
    max_so_far = 1 / smoothing * logsumexp(smoothing * comparands)
    return max_so_far, max_so_far


class STLUntimedEventually(STLFormula):
    """Represents an STL temporal operator for eventually without a time bound"""

    def __init__(self, child: STLFormula):
        """Initialize an STL formula for an untimed eventually. This formula is satisfied
        for a given signal if its child is satisfied at some point in the future

        args:
            child: an STL formula that should eventually hold
        """
        super(STLUntimedEventually, self).__init__()

        self.child = child

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed eventually operator has robustness value at any time equal to the
        maximum of its child's robustness over the rest of the signal

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Get the robustness of the child formula
        child_robustness = self.child(s, smoothing)

        # Get the robustness trace as the accumulated maximum (backwards from the
        # end of the array). See Donze et al. "Efficient Robust Monitoring for STL"
        f = lambda carry, x: accumulate_max(carry, x, smoothing)
        _, eventually_robustness = jax.lax.scan(
            f, -jnp.inf, child_robustness.x, reverse=True
        )

        return SampledSignal(s.t, eventually_robustness)


class STLTimedEventually(STLFormula):
    """Represents an STL temporal operator for eventually with a time bound"""

    def __init__(self, child: STLFormula, t_start: float, t_end: float):
        """Initialize an STL formula for an untimed eventually. This formula is satisfied
        for a given signal if its child is satisfied at some point in the future

        args:
            child: an STL formula that should eventually hold
            t_start: start point (inclusive) of interval
            t_end: end point (inclusive) of interval
        """
        super(STLTimedEventually, self).__init__()

        self.child = child

        if t_start < 0.0 or t_end <= t_start:
            raise ValueError(
                "Start time must be positive and end time must be after start"
            )

        self.t_start = t_start
        self.t_end = t_end

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        A timed eventually operator has robustness value at any time equal to the
        maximum of its child's robustness over the interval

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Get the robustness of the child formula
        child_robustness = self.child(s, smoothing)

        # We can't use the fancy moving-window maximum from Donze et al. if we want to
        # smooth the maximum (in which case all points in the interval contribute to
        # the smoothed maximum)
        i_start = 0
        i_end = -1
        T = child_robustness.T
        r = jnp.zeros((T, 1))
        for i in range(T):
            t_i = child_robustness.t[i]
            # Increase the end index until it captures the interval
            while i_end < T:
                # Move on to the next point
                i_end += 1

                # Stop once we've captured the interval, and back up if we've overshot
                if child_robustness.t[i_end] > t_i + self.t_end:
                    i_end -= 1
                    break

            # Increase the start index until it captures the interval
            while i_start < i_end and child_robustness.t[i_start] < t_i + self.t_start:
                # Move on to the next point
                i_start += 1

            # Get the maximum inside the interval
            interval = child_robustness.x[i_start : i_end + 1]

            # Pad the maximum using the interpolated value at the end time
            end_pad = child_robustness.x[i_end]
            if i_end < T - 1:
                # Get slope to interpolate
                end_slope = child_robustness.x[i_end + 1] - child_robustness.x[i_end]
                end_slope /= child_robustness.t[i_end + 1] - child_robustness.t[i_end]

                # Figure out how far into the interval the end point is
                dt = t_i + self.t_end - child_robustness.t[i_end]

                # Get the value at the end point
                end_pad += dt * end_slope

            # Do the same for the start time
            start_pad = child_robustness.x[i_start]
            if i_start > 0 and i_start < T:
                # Get slope to interpolate
                start_slope = (
                    child_robustness.x[i_start] - child_robustness.x[i_start - 1]
                )
                start_slope /= (
                    child_robustness.t[i_start] - child_robustness.t[i_start - 1]
                )

                # Figure out how far into the interval the start point is
                dt = t_i + self.t_start - child_robustness.t[i_start]

                # Get the value at the start point
                start_pad += dt * start_slope

            interval = jnp.pad(
                interval, (1, 1), "constant", constant_values=(start_pad, end_pad)
            )

            if interval.size > 0:
                interval_max = 1 / smoothing * logsumexp(smoothing * interval)
            else:
                # If interval is empty, use endpoints
                interval_max = child_robustness.x[i_start]

            r = r.at[i].set(interval_max)

        return SampledSignal(child_robustness.t, r)


class STLUntimedUntil(STLFormula):
    """Represents an STL temporal operator for until without a time bound"""

    def __init__(self, invariant: STLFormula, release: STLFormula):
        """Initialize an STL formula for an untimed until. This formula is satisfied
        for a given signal if invariant is satisfied at all times until release is
        satisfied

        args:
            invariant: an STL formula that should hold until release holds
            release: an STL formula that should hold eventually
        """
        super(STLUntimedUntil, self).__init__()

        self.invariant = invariant
        self.release = release

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed until operator has robustness value at any time t equal to

            sup_{tau in [t, infty)} min(release(tau), inf_{s in [t, tau] invariant(s)})

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        # Get the robustness of the child formulae
        invariant_robustness = self.invariant(s, smoothing)
        release_robustness = self.release(s, smoothing)

        # Do a forward pass to get the robustness of the invariant holding up to each
        # timestep (minimize by taking negative of max of negative)
        f = lambda carry, x: accumulate_max(carry, x, smoothing)
        _, r_invariant_always = jax.lax.scan(
            f, -jnp.inf, -invariant_robustness.x, reverse=False
        )
        r_invariant_always = -r_invariant_always
        invariant_always_r = SampledSignal(invariant_robustness.t, r_invariant_always)

        # Get the elementwise min with the release robustness
        r_min = -SampledSignal.max1d(
            -release_robustness, -invariant_always_r, smoothing
        )

        # Get the robustness trace as the accumulated maximum (backwards from the
        # end of the array) to compute the supremum
        _, until_robustness = jax.lax.scan(f, -jnp.inf, r_min.x, reverse=True)

        return SampledSignal(r_min.t, until_robustness)


class STLUntimedAlways(STLFormula):
    """Represents an STL temporal operator for always without a time bound"""

    def __init__(self, child: STLFormula):
        """Initialize an STL formula for an untimed always. This formula is satisfied
        for a given signal if its child is satisfied at all points in the future.

        Implemented as not eventually not child.

        args:
            child: an STL formula that should always hold
        """
        super(STLUntimedAlways, self).__init__()

        self.child = STLNegation(STLUntimedEventually(STLNegation(child)))

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed always operator is equivalent to not (eventually not child)

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        return self.child(s, smoothing)


class STLTimedAlways(STLFormula):
    """Represents an STL temporal operator for always with a time bound"""

    def __init__(self, child: STLFormula, t_start: float, t_end: float):
        """Initialize an STL formula for an untimed always. This formula is satisfied
        for a given signal if its child is satisfied at all points in the future.

        Implemented as not eventually not child.

        args:
            child: an STL formula that should always hold
            t_start: start point (inclusive) of interval
            t_end: end point (inclusive) of interval
        """
        super(STLTimedAlways, self).__init__()

        self.child = STLNegation(STLTimedEventually(STLNegation(child), t_start, t_end))

    def __call__(self, s: SampledSignal, smoothing: float = 100.0) -> SampledSignal:
        """Evaluates this formula on the given signal, returning its robustness trace.

        The robustness trace is a SampledSignal of the same length as s where each
        sample represents the robustness of the subsignal of s starting at that sample
        and continuing to the end of s. I.e. if the robustness trace has sample
        (t_i, r_i), then the subsignal s[i:] has robustness r_i with respect to this
        formula.

        If robustness is positive, then the formula is satisfied. If the robustness
        is negative, then the formula is not satisfied.

        An untimed always operator is equivalent to not (eventually not child)

        args:
            s: the signal upon which this formula is evaluated.
            smoothing: the parameter determining how much to smooth non-continuous
                elements of this formula (e.g. mins and maxes). Uses the log-sum-exp
                smoothing for max and min.
        returns:
            SampledSignal of the robustness trace for this formula and the given signal.
        """
        return self.child(s, smoothing)
