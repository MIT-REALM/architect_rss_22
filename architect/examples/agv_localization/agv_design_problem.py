import jax.numpy as jnp

from architect.design import DesignProblem
from .agv_cost import agv_cost, agv_max_estimation_error
from .agv_design_parameters import AGVDesignParameters
from .agv_exogenous_parameters import AGVExogenousParameters
from .agv_simulator import agv_simulate


def make_agv_localization_design_problem(T: float, dt: float) -> DesignProblem:
    """Make an instance of the AGV localization and navigation design problem.

    args:
        T: time to simulate
        dt: timestep
    returns:
        a DesignProblem
    """
    time_steps = int(T / dt)

    # Define the exogenous parameters
    observation_noise_covariance = jnp.diag(jnp.array([0.1, 0.01, 0.01]))
    actuation_noise_covariance = dt ** 2 * jnp.diag(jnp.array([0.001, 0.001, 0.01]))
    initial_state_mean = jnp.array([-2.2, 0.5, 0.0])
    initial_state_covariance = 0.001 * jnp.eye(3)
    ep = AGVExogenousParameters(
        time_steps,
        initial_state_mean,
        initial_state_covariance,
        actuation_noise_covariance,
        observation_noise_covariance,
    )

    # Define the design parameters
    beacon_locations = jnp.array([[-2.0, 0.0], [-0.1, 0.0]])
    control_gains = jnp.array([0.5, 0.1])
    dp = AGVDesignParameters()
    dp.set_values(jnp.concatenate((control_gains, beacon_locations.reshape(-1))))

    # Wrap the cost function
    def cost_fn(design_params, exogenous_sample):
        return agv_cost(
            design_params,
            exogenous_sample,
            observation_noise_covariance,
            actuation_noise_covariance,
            initial_state_mean,
            initial_state_covariance,
            time_steps,
            dt,
        )

    # Wrap the simulator function
    def simulator(design_params, exogenous_sample):
        return agv_simulate(
            design_params,
            exogenous_sample,
            observation_noise_covariance,
            actuation_noise_covariance,
            initial_state_mean,
            initial_state_covariance,
            time_steps,
            dt,
        )

    # Make a design problem instance
    agv_design_problem = DesignProblem(dp, ep, cost_fn, simulator)
    return agv_design_problem


def make_agv_localization_design_problem_analysis(T: float, dt: float) -> DesignProblem:
    """Make an instance of the AGV localization and navigation design problem.

    Uses the maximum estimation error instead of the mixed cost

    args:
        T: time to simulate
        dt: timestep
    returns:
        a DesignProblem
    """
    time_steps = int(T / dt)

    # Define the exogenous parameters
    observation_noise_covariance = jnp.diag(jnp.array([0.1, 0.01, 0.01]))
    actuation_noise_covariance = dt ** 2 * jnp.diag(jnp.array([0.001, 0.001, 0.01]))
    initial_state_mean = jnp.array([-2.2, 0.5, 0.0])
    initial_state_covariance = 0.001 * jnp.eye(3)
    ep = AGVExogenousParameters(
        time_steps,
        initial_state_mean,
        initial_state_covariance,
        actuation_noise_covariance,
        observation_noise_covariance,
    )

    # Define the design parameters
    beacon_locations = jnp.array([[-2.0, 0.0], [-0.1, 0.0]])
    control_gains = jnp.array([0.5, 0.1])
    dp = AGVDesignParameters()
    dp.set_values(jnp.concatenate((control_gains, beacon_locations.reshape(-1))))

    # Wrap the cost function
    def cost_fn(design_params, exogenous_sample):
        return agv_max_estimation_error(
            design_params,
            exogenous_sample,
            observation_noise_covariance,
            actuation_noise_covariance,
            initial_state_mean,
            initial_state_covariance,
            time_steps,
            dt,
        )

    # Wrap the simulator function
    def simulator(design_params, exogenous_sample):
        return agv_simulate(
            design_params,
            exogenous_sample,
            observation_noise_covariance,
            actuation_noise_covariance,
            initial_state_mean,
            initial_state_covariance,
            time_steps,
            dt,
        )

    # Make a design problem instance
    agv_design_problem = DesignProblem(dp, ep, cost_fn, simulator)
    return agv_design_problem
