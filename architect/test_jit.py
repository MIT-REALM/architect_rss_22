import time

import jax
import jax.numpy as jnp


def f(x):
    return jnp.dot(x, x.T)


N_trials = 1
total_time = 0.0
x = jnp.ones((1000, 1))

f = jax.jit(f)
start = time.perf_counter()
f(x)
end = time.perf_counter()
print(f"jit_time: {end - start}")

g = jax.vmap(f)
y = jnp.ones((1000, 1000, 1))
start = time.perf_counter()
g(y)
end = time.perf_counter()
print(f"jit_time: {end - start}")

y = jnp.ones((1001, 1000, 1))
start = time.perf_counter()
g(y)
end = time.perf_counter()
print(f"jit_time: {end - start}")


for i in range(N_trials):
    start = time.perf_counter()
    g(y)
    end = time.perf_counter()
    total_time += end - start

print(f"total_time: {total_time}")
