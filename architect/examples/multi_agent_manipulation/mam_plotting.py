import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from celluloid import Camera


def make_box_patches(
    box_state, alpha: float, box_side_length: float, ax, hatch: bool = False
):
    """Adds patches for visualizing the box to the given axes

    args:
        box_state: (x, y, theta, vx, vy, thetadot)
        alpha: float transparency
        box_side_length: float side length of box
        ax: matplotlib axes
        hatch: if True, hatch the box patch
    returns:
        a list of properly transformed and colored patches for the box
    """
    patches_list = []

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
        hatch=("/" if hatch else None),
    )
    ax.add_patch(box)
    patches_list.append(box)

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
    patches_list.append(arrow)

    return patches_list


def make_turtle_patches(turtle_state, alpha: float, radius: float, ax):
    """Adds patches for visualizing the turtle to the given axes

    args:
        turtle_state: (x, z, theta, vx, vz, thetadot)
        alpha: float transparency
        radius: float radius of turtlebot
        ax: matplotlib axes
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
    alphas = jnp.linspace(0.3, 1.0, n_steps)
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
    alphas = jnp.linspace(0.3, 1.0, n_steps)
    for i in i_to_show:
        make_box_patches(box_states[i], alphas[i].item(), box_size, ax)


def make_pushing_animation(
    box_states,
    turtle_states,
    desired_box_pose,
    box_size: float,
    radius: float,
    n_steps_to_show: int,
    ms_per_frame: int,
    save_filename: str,
):
    """Make an animation of the pushing action and save it

    args:
        box_states: (N, 6) array of box states
        turtle_states: (N, n_turtles, 6) array of turtlebot states
        desired_box_pose: (3,) array of (x, y, theta) desired box pose
        box_size: float box_size of turtlebot
        radius: float turtlebot radius
        n_steps_to_show: plot a continuous line for the trajectory along with
                         `n_steps_to_show` circles for the turtlebot at different points
                         in time
        ms_per_frame: milliseconds per frame
        save_filename: filename where the animation should be saved.
    """
    # Make a figure for the animation
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    camera = Camera(fig)

    # For each frame, plot the turtlebots and box
    n_steps = box_states.shape[0]
    n_turtles = turtle_states.shape[1]
    i_to_show = jnp.linspace(0, n_steps, n_steps_to_show, dtype=int)
    for i in i_to_show:
        # Plot box center-of-mass trajectory
        ax.plot(
            box_states[:i, 0],
            box_states[:i, 1],
            color=plt.get_cmap("Blues")(1.0),
        )

        # Plot box patch
        make_box_patches(box_states[i], 1.0, box_size, ax)

        # Plot desired box pose
        make_box_patches(desired_box_pose, 1.0, box_size, plt.gca(), hatch=True)
        label = "Desired box pose" if i == i_to_show[0] else None
        ax.fill_between(
            [],
            [],
            [],
            edgecolor=plt.get_cmap("Blues")(1.0),
            hatch="xx",
            label=label,
            facecolor="none",
        )
        ax.legend()

        for j in range(n_turtles):
            # Plot turtle center-of-mass trajectory
            ax.plot(
                turtle_states[:i, j, 0],
                turtle_states[:i, j, 1],
                color=plt.get_cmap("Oranges")(1.0),
            )

            # Plot turtle patch
            make_turtle_patches(turtle_states[i, j], 1.0, radius, ax)

        # Prettify
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([-0.75, 1.0])
        plt.ylim([-0.75, 1.0])
        plt.gca().set_aspect("equal")

        # Take a snapshot
        camera.snap()

    # Save the animation
    animation = camera.animate(interval=ms_per_frame)
    animation.save(save_filename)
