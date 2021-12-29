import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms


def make_box_patches(box_state, alpha: float, box_side_length: float, ax):
    """Adds patches for visualizing the box to the given axes

    args:
        box_state: (x, y, theta, vx, vy, thetadot)
        alpha: float transparency
        box_side_length: float side length of box
        ax: matplotlib axes
    returns:
        a list of properly transformed and colored patches for the box
    """
    box_xy = box_state[:2]
    box_theta = box_state[2]
    xform = transforms.Affine2D()
    xform = xform.rotate_around(
        box_side_length / 2.0, box_side_length / 2.0, theta=box_theta
    )
    xform = xform.translate(*(box_xy - box_side_length / 2.0))
    xform = xform + ax.transData
    box = patches.Rectangle(
        (0, 0),
        box_side_length,
        box_side_length,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Blues")(0.1 + alpha),
        fill=False,
    )
    ax.add_patch(box)

    # Add an arrow pointing up
    xform = transforms.Affine2D()
    xform = xform.rotate_around(0.0, 0.0, theta=box_theta - jnp.pi / 2.0)
    xform = xform.translate(*box_xy)
    xform = xform + ax.transData
    arrow = patches.Arrow(
        0,
        0,
        0,
        box_side_length / 8,
        width=box_side_length / 20,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Blues")(0.1 + alpha),
        facecolor=plt.get_cmap("Blues")(0.1 + alpha),
        fill=True,
    )
    ax.add_patch(arrow)


def make_turtle_patches(turtle_state, alpha: float, radius: float, ax):
    """Adds patches for visualizing the turtle to the given axes

    args:
        turtle_state: (x, z, theta, vx, vz, thetadot)
        alpha: float transparency
        radius: float radius of turtlebot
        ax: matplotlib axes
    returns:
        a list of properly transformed and colored patches for the box
    """
    turtle_xy = turtle_state[:2]
    xform = transforms.Affine2D()
    xform = xform.translate(*turtle_xy)
    xform = xform + ax.transData
    turtle = patches.Circle(
        (0, 0),
        radius,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Oranges")(0.1 + alpha),
        fill=False,
    )
    ax.add_patch(turtle)

    # Add an arrow pointing up
    turtle_theta = turtle_state[2]
    xform = transforms.Affine2D()
    xform = xform.rotate_around(0.0, 0.0, theta=turtle_theta - jnp.pi / 2.0)
    xform = xform.translate(*turtle_xy)
    xform = xform + ax.transData
    arrow = patches.Arrow(
        0,
        0,
        0,
        0.8 * radius,
        width=radius / 2,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Oranges")(0.1 + alpha),
        facecolor=plt.get_cmap("Oranges")(0.1 + alpha),
        fill=True,
    )
    ax.add_patch(arrow)


def plot_turtle_trajectory(turtle_states, radius: float, n_steps_to_show: int, ax):
    """Plot a trajectory of the turtlebot on the given axes.

    args:
        turtle_states: (N, 6) array of states
        radius: float radius of turtlebot
        n_steps_to_show: plot a continuous line for the trajectory along with
                         `n_steps_to_show` circles for the turtlebot at different points
                         in time
        ax: the matplotlib axis to plot upon
    """
    # Plot the center-of-mass trajectory
    ax.plot(
        turtle_states[:, 0],
        turtle_states[:, 1],
        label="Turtlebot",
        color=plt.get_cmap("Oranges")(1.0),
    )

    # Draw the snapshots
    n_steps = turtle_states.shape[0]
    i_to_show = jnp.linspace(0, n_steps, n_steps_to_show, dtype=int)
    alphas = jnp.linspace(0.0, 1.0, n_steps)
    for i in i_to_show:
        make_turtle_patches(turtle_states[i], alphas[i].item(), radius, ax)


def plot_box_trajectory(box_states, box_size: float, n_steps_to_show: int, ax):
    """Plot a trajectory of the turtlebot on the given axes.

    args:
        box_states: (N, 6) array of states
        box_size: float box_size of turtlebot
        n_steps_to_show: plot a continuous line for the trajectory along with
                         `n_steps_to_show` circles for the turtlebot at different points
                         in time
        ax: the matplotlib axis to plot upon
    """
    # Plot the center-of-mass trajectory
    ax.plot(
        box_states[:, 0],
        box_states[:, 1],
        label="Box",
        color=plt.get_cmap("Blues")(1.0),
    )

    # Draw the snapshots
    n_steps = box_states.shape[0]
    i_to_show = jnp.linspace(0, n_steps, n_steps_to_show, dtype=int)
    alphas = jnp.linspace(0.0, 1.0, n_steps)
    for i in i_to_show:
        make_box_patches(box_states[i], alphas[i].item(), box_size, ax)
