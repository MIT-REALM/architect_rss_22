"""Define 2D transforms (rotations and translations)"""
import jax
import jax.numpy as jnp


@jax.jit
def rotation_matrix_2d(theta: jnp.ndarray) -> jnp.ndarray:
    """Return the matrix for rotating by theta radians (positive = counterclockwise)
    around the z axis in the x-y plane

    args:
        theta: 1-element, 0-dimensional array containing the angle to rotate around
    returns: a (2, 2) rotation matrix
    """
    # Construct the rotation matrix
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    return R
