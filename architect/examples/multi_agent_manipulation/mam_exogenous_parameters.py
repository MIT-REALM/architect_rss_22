from typing import Optional

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

from architect.design import ExogenousParameters


class MAMExogenousParameters(ExogenousParameters):
    """ExogenousParameters for the multi-agent manipulation task"""

    def __init__(
        self,
        mu_turtle_ground_range: jnp.ndarray,
        mu_box_ground_range: jnp.ndarray,
        mu_box_turtle_range: jnp.ndarray,
        box_mass_range: jnp.ndarray,
        desired_box_pose_range: jnp.ndarray,
        turtlebot_displacement_covariance: jnp.ndarray,
        n_turtles: int,
    ):
        """
        Initialize the exogenous parameters for the multi-agent manipulation problem.

        args:
            mu_turtle_ground_range: (min, max) values for friction coefficient between
                turtlebot and ground.
            mu_box_ground_range: (min, max) values for friction coefficient between box
                and ground.
            mu_box_turtle_range: (min, max) values for friction coefficient between box
                and turtlebot.
            box_mass_range: (min, max) values for box mass
            desired_box_pose_range: (3, 2) array of (min, max) values for desired x, y,
                and theta for the box.
            turtlebot_displacement_covariance: (3, 3) covariance matrix for displacement
                from nominal initial position of turtlebot.
            n_turtles: number of turtlebots
        """
        n_vars = 7 + 3 * n_turtles
        names = ["mu_turtle_ground", "mu_box_ground", "mu_box_turtle", "box_mass"]
        names += ["box_desired_x", "box_desired_y", "box_desired_theta"]
        for i in range(n_turtles):
            names += [
                f"turtle_{i}_initial_displacement_x",
                f"turtle_{i}_initial_displacement_y",
                f"turtle_{i}_initial_displacement_theta",
            ]

        self.n_turtles = n_turtles
        self.mu_turtle_ground_range = mu_turtle_ground_range
        self.mu_box_ground_range = mu_box_ground_range
        self.mu_box_turtle_range = mu_box_turtle_range
        self.box_mass_range = box_mass_range
        self.desired_box_pose_range = desired_box_pose_range
        self.turtlebot_displacement_covariance = turtlebot_displacement_covariance

        super(MAMExogenousParameters, self).__init__(n_vars, names)

    def sample(
        self, prng_key: PRNGKeyArray, batch_size: Optional[int] = None
    ) -> jnp.ndarray:
        """Sample values for these exogenous parameters from this distribution.

        args:
            prng_key: a 2-element JAX array containing the PRNG key used for sampling.
                      This method will not split the key, it will be consumed.
            batch_size: if None (default), return a 1D JAX array with self.size
                        elements; otherwise, return a 2D JAX array with size
                        (batch_size, self.size)
        """
        if batch_size is None:
            batch_size = 1

        sampled_phi = jnp.zeros((batch_size, self.size))

        # Sample friction coefficients
        prng_key, subkey = jax.random.split(prng_key)
        mu_turtle_ground = jax.random.uniform(
            subkey,
            minval=self.mu_turtle_ground_range[0],
            maxval=self.mu_turtle_ground_range[1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 0].set(mu_turtle_ground)
        mu_box_ground = jax.random.uniform(
            subkey,
            minval=self.mu_box_ground_range[0],
            maxval=self.mu_box_ground_range[1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 1].set(mu_box_ground)
        mu_box_turtle = jax.random.uniform(
            subkey,
            minval=self.mu_box_turtle_range[0],
            maxval=self.mu_box_turtle_range[1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 2].set(mu_box_turtle)

        # Sample box mass
        box_mass = jax.random.uniform(
            subkey,
            minval=self.box_mass_range[0],
            maxval=self.box_mass_range[1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 3].set(box_mass)

        # Sample desired box position
        desired_box_x = jax.random.uniform(
            subkey,
            minval=self.desired_box_pose_range[0, 0],
            maxval=self.desired_box_pose_range[0, 1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 4].set(desired_box_x)
        desired_box_y = jax.random.uniform(
            subkey,
            minval=self.desired_box_pose_range[1, 0],
            maxval=self.desired_box_pose_range[1, 1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 5].set(desired_box_y)
        desired_box_x = jax.random.uniform(
            subkey,
            minval=self.desired_box_pose_range[2, 0],
            maxval=self.desired_box_pose_range[2, 1],
            shape=(batch_size,),
        )
        sampled_phi = sampled_phi.at[:, 6].set(desired_box_x)

        # Sample turtlebot displacements
        for i in range(self.n_turtles):
            prng_key, subkey = jax.random.split(prng_key)
            turtle_displacement = jax.random.multivariate_normal(
                subkey,
                jnp.zeros(3),
                self.turtlebot_displacement_covariance,
                shape=(batch_size,),
            )
            sampled_phi = sampled_phi.at[:, 7 + i * 3 : 7 + (i + 1) * 3].set(
                turtle_displacement
            )

        # Remove batch dimension if we only have one batch
        if batch_size == 1:
            sampled_phi = sampled_phi.reshape(-1)

        return sampled_phi
