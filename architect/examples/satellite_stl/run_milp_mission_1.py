from math import sqrt
import time

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp


from architect.optimization import (
    AdversarialLocalOptimizer,
)
from architect.examples.satellite_stl.sat_design_problem import (
    make_sat_design_problem,
)
from architect.examples.satellite_stl.sat_stl_specification import (
    make_sat_rendezvous_specification,
)


def L1Norm(model, x):
    # From Dawei's STL planning code
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == gp.abs_(xvar[i]))
    return sum(abs_x)


def main(seed: int):
    # Problem parameters
    t_sim = 200.0
    dt = 2.0
    time_steps = int(t_sim // dt)
    n_dims = 6
    n_controls = 3
    speed_threshold = 0.1  # m/s
    docking_threshold = 0.1  # m
    min_waiting_radius = 2.0  # m

    # Other dynamics parameters
    MU = 3.986e14  # Earth's gravitational parameter (m^3 / s^2)
    A_LEO = 353e3  # GEO semi-major axis (m)
    M_CHASER = 500  # chaser satellite mass
    N = sqrt(MU / A_LEO ** 3)  # mean-motion parameter
    BIGM = 1e3  # for binary logic constraints

    # Create a new model
    m = gp.Model("mission_1")
    # m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.MIPGap, 1)

    # Create a trajectory optimization problem with states, reference states, control
    # inputs, and feedback matrix
    x = m.addMVar((time_steps, n_dims), lb=-100.0, ub=100.0, name="x")
    x_ref = m.addMVar((time_steps, n_dims), lb=-100.0, ub=100.0, name="x_ref")
    u = m.addMVar((time_steps, n_controls), lb=-100.0, ub=100.0, name="u")
    u_ref = m.addMVar((time_steps, n_controls), lb=-100.0, ub=100.0, name="u_ref")

    # We cannot simultaneously optimize K, so set it to be something reasonable
    K = np.array(
        [
            [27.995287, -5.337199, -0.6868453, 72.93794, 3.8204267, 0.19549589],
            [3.8033, 25.740, -2.8299, -2.809, 72.12, -0.5243],
            [3.2838054, -1.0270333, 24.277672, 0.7584929, -0.85453165, 71.199554],
        ]
    )

    # Add an objective for the total impulse required for the manuever
    impulses = [dt * L1Norm(m, [ui for ui in u[t]]) for t in range(time_steps)]
    total_effort = sum(impulses)
    m.setObjective(total_effort, GRB.MINIMIZE)

    # Initial condition constraints
    m.addConstr(x[0, 0] == 11.5, "px_0")
    m.addConstr(x[0, 1] == 11.5, "py_0")
    m.addConstr(x[0, 2] == 0.0, "pz_0")
    m.addConstr(x[0, 3] == 0.0, "vx_0")
    m.addConstr(x[0, 4] == 0.0, "vy_0")
    m.addConstr(x[0, 5] == 0.0, "vz_0")

    # Encode the dynamics in a simple discrete-time form
    for t in range(1, time_steps):
        # Control law
        m.addConstr(
            u[t - 1, :] == u_ref[t - 1, :] - K @ x[t - 1, :] - K @ x_ref[t - 1, :],
            f"u({t})",
        )

        # CHW dynamics
        m.addConstr(x[t, 0] == x[t - 1, 0] + dt * x[t - 1, 3], f"d(px)/dt({t})")
        m.addConstr(x[t, 1] == x[t - 1, 1] + dt * x[t - 1, 4], f"d(py)/dt({t})")
        m.addConstr(x[t, 2] == x[t - 1, 2] + dt * x[t - 1, 5], f"d(pz)/dt({t})")
        m.addConstr(
            x[t, 3]
            == x[t - 1, 3]
            + dt
            * (3 * N ** 2 * x[t - 1, 0] + 2 * N * x[t - 1, 4] + u[t - 1, 0] / M_CHASER),
            f"d(vx)/dt({t})",
        )
        m.addConstr(
            x[t, 4]
            == x[t - 1, 4] + dt * (-2 * N * x[t - 1, 3] + u[t - 1, 1] / M_CHASER),
            f"d(vy)/dt({t})",
        )
        m.addConstr(
            x[t, 5]
            == x[t - 1, 5] + dt * (-(N ** 2) * x[t - 1, 2] + u[t - 1, 2] / M_CHASER),
            f"d(vz)/dt({t})",
        )

    # Add the "eventually reach the target" constraint.
    # Start by encoding the robustness of the docking distance predicate, and then
    # encode the eventually robustness at each timestep as being the max over
    # remaining timesteps
    r_goal = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_goal")
    for t in range(time_steps):
        distance_to_goal_t = L1Norm(m, [xi for xi in x[t, :3]])
        m.addConstr(r_goal[t] == docking_threshold - distance_to_goal_t)

    r_f_goal = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_f_goal")
    m.addConstr(r_f_goal[time_steps - 1] == r_goal[time_steps - 1])
    for t in reversed(range(time_steps - 1)):
        m.addConstr(r_f_goal[t] == gp.max_([r_goal[t], r_f_goal[t + 1], -BIGM]))

    # Require that the eventually robustness is positive at the start of the trace
    m.addConstr(r_f_goal[0] >= 0)

    # Now add robustness traces for being close to the target and being slow
    r_close = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_close")
    r_not_close = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_close")
    r_slow = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_slow")
    for t in range(time_steps):
        distance_to_goal_t = L1Norm(m, [xi for xi in x[t, :3]])
        m.addConstr(r_close[t] == min_waiting_radius - distance_to_goal_t)
        m.addConstr(r_not_close[t] == -r_close[t])

        speed_t = L1Norm(m, [xi for xi in x[t, 3:]])
        m.addConstr(r_slow[t] == speed_threshold - speed_t)

    # Make a robustness trace for "always slow"
    r_g_slow = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_slow")
    m.addConstr(r_g_slow[time_steps - 1] == r_slow[time_steps - 1])
    for t in reversed(range(time_steps - 1)):
        m.addConstr(r_g_slow[t] == gp.min_([r_slow[t], r_g_slow[t + 1], BIGM]))

    # Make a robustness trace for "not close until always slow".
    # This requires a trace for "always not close until this time"
    r_not_close_until = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_slow")
    m.addConstr(r_not_close_until[0] == r_not_close[0])
    for t in range(1, time_steps):
        m.addConstr(
            r_not_close_until[t]
            == gp.min_([r_not_close[t], r_not_close_until[t - 1], BIGM])
        )

    # Make a robustness trace for the until happening at each timestep
    r_until_here = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_slow")
    for t in range(time_steps):
        m.addConstr(
            r_until_here[t] == gp.min_([r_not_close_until[t], r_g_slow[t], BIGM])
        )

    # The until robustness is the maximum of r_until_here over the rest of the trace
    r_until = m.addVars(time_steps, lb=-BIGM, ub=BIGM, name="r_until")
    m.addConstr(r_until[time_steps - 1] == r_until_here[time_steps - 1])
    for t in reversed(range(time_steps - 1)):
        m.addConstr(r_until[t] == gp.max_([r_until_here[t], r_until[t + 1], -BIGM]))

    # Require that the until robustness is positive at the start of the trace
    m.addConstr(r_until[0] >= 0)

    # Solve the problem
    start = time.time()
    m.optimize()
    end = time.time()
    mip_solve_time = end - start
    print("solving MIP takes %.3f s" % (mip_solve_time))

    # Extract the design parameters
    x_ref_opt = x_ref.X
    u_ref_opt = u_ref.X
    planned_trajectory = np.vstack((u_ref_opt, x_ref_opt))
    design_params_np = np.concatenate((K.reshape(-1), planned_trajectory.reshape(-1)))
    dp_opt = jnp.array(design_params_np)

    prng_key = jax.random.PRNGKey(seed)

    # Make the design problem
    t_sim = 200.0
    dt = 2.0
    time_steps = int(t_sim // dt)
    specification_weight = 2e4
    prng_key, subkey = jax.random.split(prng_key)
    sat_design_problem = make_sat_design_problem(specification_weight, time_steps, dt)

    # Create the optimizer
    ad_opt = AdversarialLocalOptimizer(sat_design_problem)

    # Run a simulation for plotting the optimal solution with the original exogenous
    # parameters
    ep_opt = jnp.ndarray([11.5, 11.5, 0, 0, 0, 0])
    state_trace, total_effort = sat_design_problem.simulator(dp_opt, ep_opt)

    # Get the robustness of this solution
    stl_specification = make_sat_rendezvous_specification()
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0])
    signal = jnp.vstack((t.reshape(1, -1), state_trace.T))
    original_robustness = stl_specification(signal)

    original_cost = -original_robustness[1, 0] + total_effort / specification_weight

    # Do the adversarial optimization
    sat_design_problem.design_params.set_values(dp_opt)

    # x_opt = x.X
    # ax = plt.axes(projection="3d")
    # ax.plot3D(x_opt[:, 0], x_opt[:, 1], x_opt[:, 2])
    # ax.plot3D(0.0, 0.0, 0.0, "ko")
    # ax.plot3D(x_opt[0, 0], x_opt[0, 1], x_opt[0, 2], "ks")
    # plt.show()

    # r_goals = []
    # r_f_goals = []
    # for t in range(time_steps):
    #     r_goals.append(r_goal[t].X)
    #     r_f_goals.append(r_f_goal[t].X)

    # plt.plot(r_goals, label="goal")
    # plt.plot(r_f_goals, label="eventually goal")
    # plt.legend()
    # plt.show()

    # r_not_closes = []
    # r_slows = []
    # r_g_slows = []
    # r_untils = []
    # r_not_close_untils = []
    # for t in range(time_steps):
    #     r_not_closes.append(r_not_close[t].X)
    #     r_slows.append(r_slow[t].X)
    #     r_g_slows.append(r_g_slow[t].X)
    #     r_not_close_untils.append(r_not_close_until[t].X)
    #     r_untils.append(r_until[t].X)

    # plt.plot(r_not_closes, label="not_close")
    # plt.plot(r_slows, label="slow")
    # plt.plot(r_g_slows, label="always slow")
    # plt.plot(r_not_close_untils, label="not close until t")
    # plt.plot(r_untils, label="not close until always slow")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
