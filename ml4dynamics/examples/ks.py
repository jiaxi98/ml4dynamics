r"""
Examples with Kuramoto–Sivashinsky equation

$$
  u_t + (c + u)u_x + u_{xx} + \nu u_{xxxx} = 0.
$$

We perform sensitivity analysis of several statistics of the KS equation
w.r.t. the parameter c. We reproduce the fig. 1 in reference [1]. This
statistics can also be used to evaluate the ROM we proposed. To reproduce the
figure, we use the following parameters in config file:

ks:
  nu: 1
  c: 0.8
  L: 128
  T: 400
  init_scale: 1.
  nx: 1024
  dt: .1
  BC: Dirichlet-Neumann

reference:
1. https://arxiv.org/pdf/1307.8197
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import yaml
from box import Box
from jax import random as random
from matplotlib import pyplot as plt

from ml4dynamics.dynamics import KS

with open("config/simulation.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
# model parameters
nu = config.ks.nu
c = config.ks.c
L = config.ks.L
T = config.ks.T
init_scale = config.ks.init_scale
# solver parameters
N = config.ks.nx
dt = config.ks.dt
n_sample = 10
rng = random.PRNGKey(config.sim.seed)

# KS simulator with Dirichlet Neumann BC
ks1 = KS(
  N=N - 1,
  T=T,
  dt=dt,
  init_scale=init_scale,
  L=L,
  nu=nu,
  c=c,
  BC="Dirichlet-Neumann",
  rng=rng,
)
ks2 = KS(
  N=N // 2 - 1,
  T=T,
  dt=dt,
  init_scale=init_scale,
  L=L,
  nu=nu,
  c=c,
  BC="Dirichlet-Neumann",
  rng=rng,
)
ks3 = KS(
  N=N // 4 - 1,
  T=T,
  dt=dt,
  init_scale=init_scale,
  L=L,
  nu=nu,
  c=c,
  BC="Dirichlet-Neumann",
  rng=rng,
)
ks_models = [ks1, ks2, ks3]

c_array = jnp.linspace(0.0, 2.0, 21)
# NOTE: one can use this script to test the change of accuracy in grid
# coarsening
# c_array = jnp.array([0.8])
dx = L / N
x = jnp.linspace(dx, L - dx, N - 1)
ubar = jnp.zeros((3, c_array.shape[0]))
u_std = jnp.zeros((3, c_array.shape[0]))
u2bar = jnp.zeros((3, c_array.shape[0]))
u2_std = jnp.zeros((3, c_array.shape[0]))
color_array = ["r", "b", "g"]

for i in range(c_array.shape[0]):
  print(f"{c_array[i]:.2f}")
  umean_tmp = jnp.zeros((3, n_sample))
  u2mean_tmp = jnp.zeros((3, n_sample))
  fig, axs = plt.subplots(1, 2, figsize=(9, 3))
  for _ in range(n_sample):
    rng, key = random.split(rng)
    r = random.uniform(key) * 20 + 44
    u0 = jnp.exp(-(x - r)**2 / r**2 * 4)
    # u0 = random.uniform(key) * jnp.sin(8 * jnp.pi * x / 128) +\
    #   random.uniform(rng) * jnp.sin(16 * jnp.pi * x / 128)
    for ks in ks_models:
      ks.c = c_array[i]
      ks.assembly_matrix()

    ks_models[0].run_simulation(u0, ks_models[0].CN_FEM)
    ks_models[1].run_simulation(u0[1::2], ks_models[1].CN_FEM)
    ks_models[2].run_simulation(u0[3::4], ks_models[2].CN_FEM)
    for j in range(len(ks_models)):
      umean_tmp = umean_tmp.at[j, _].set(jnp.mean(ks_models[j].x_hist[-500:]))
      u2mean_tmp = u2mean_tmp.at[j, _].set(
        jnp.mean(ks_models[j].x_hist[-500:]**2)
      )
      # axs[0].scatter(c_array[i], umean, c=color_array[j], s=2)
      # axs[1].scatter(c_array[i], u2mean, c=color_array[j], s=2)
      if n_sample <= 2:
        axs[0].plot(
          jnp.arange(ks_models[j].x_hist.shape[0]) * dt,
          jnp.mean(ks_models[j].x_hist, axis=1)
        )
        axs[1].plot(
          jnp.arange(ks_models[j].x_hist.shape[0]) * dt,
          jnp.mean(ks_models[j].x_hist**2, axis=1)
        )
  ubar = ubar.at[:, i].set(jnp.mean(umean_tmp, axis=1))
  u2bar = u2bar.at[:, i].set(jnp.mean(u2mean_tmp, axis=1))
  u_std = u_std.at[:, i].set(jnp.std(umean_tmp, axis=1))
  u2_std = u2_std.at[:, i].set(jnp.std(u2mean_tmp, axis=1))
  print(u2bar[:, i])
  if n_sample <= 2:
    plt.savefig(f"results/fig/hist{c_array[i]:.2f}.png")
  plt.close()
_, axs = plt.subplots(1, 2, figsize=(10, 4))
for j in range(len(ks_models)):
  axs[0].fill_between(
    c_array,
    ubar[j] - u_std[j],
    ubar[j] + u_std[j],
    label=r"$N = {}$".format(N // (2**j)),
    color=color_array[j],
    alpha=0.5
  )
  axs[1].fill_between(
    c_array,
    u2bar[j] - u2_std[j],
    u2bar[j] + u2_std[j],
    label=r"$N = {}$".format(N // (2**j)),
    color=color_array[j],
    alpha=0.5
  )
axs[0].set_xlabel(r"$c$")
axs[1].set_xlabel(r"$c$")
axs[0].set_ylabel(r"$\langle \overline{u} \rangle$")
axs[1].set_ylabel(r"$\langle \overline{u^2} \rangle$")
axs[0].legend()
axs[1].legend()
plt.savefig("results/fig/ks_c_stats.png")
