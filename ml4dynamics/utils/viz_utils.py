import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation

from ml4dynamics.utils import calc_utils


def plot_psd_cmp(u_list, dx_list, title_list, fig_name):
  """Compare and plot the power spectrum density of the fluid field"""

  if len(u_list) != len(dx_list) or len(u_list) != len(title_list):
    breakpoint()
    raise ValueError("The length of u_list and dx_list must be the same.")
  for i in range(len(u_list)):
    k_bins, E_k_avg = calc_utils.power_spec_over_t(u_list[i], dx_list[i])
    # assert np.linalg.norm(k_bins - k_bins_true) < 1e-14
    plt.plot(k_bins, E_k_avg, label=title_list[i])
    if i == 0:
      plt.plot(
        k_bins, E_k_avg[1] * (k_bins / k_bins[1])**(-5 / 3), label="-5/3 law"
      )
  plt.xlabel("k")
  plt.ylabel("E(k)")
  plt.xscale("log")
  plt.yscale("log")
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.title("2D Power Spectrum")
  plt.legend()
  plt.savefig(f"results/fig/psd_{fig_name}.png")
  plt.close()


def plot_from_h5py(case: str):
  """Plot the fluid field from the h5py file"""

  path = "data/RD/" + case + ".h5"
  with h5py.File(path, "r") as file:
    U = file["input"][:]
    case_num, nt, dim, nx, ny = U.shape
    T = file["end_time"][()]
    dt = T / nt

  n1 = 5
  n2 = 5

  for k in range(10):
    random_integer = np.random.randint(1, 101)
    print(random_integer)
    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(
          U[random_integer, (i * n2 + j) * 30, 0, :, :], cmap=cm.twilight
        )
        plt.axis("off")
    plt.savefig("results/fig/traj_" + str(random_integer) + ".pdf")


def plot_from_h5py_cmp():
  """Plot the fluid field from the h5py file"""

  path = "data/RD/128-100.h5"
  with h5py.File(path, "r") as file:
    U128 = file["input"][:]

  path = "data/RD/64-100.h5"
  with h5py.File(path, "r") as file:
    U64 = file["input"][:]

  n1 = 5
  n2 = 5

  for k in range(10):
    random_integer = np.random.randint(1, 101)
    print(random_integer)
    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(
          U128[random_integer, (i * n2 + j) * 30, 0, :, :], cmap=cm.viridis
        )
        plt.axis("off")
    plt.savefig("results/fig/traj_128_" + str(random_integer) + ".pdf")
    plt.close()

    plt.figure(figsize=(20, 20))
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n2, n1, i * n2 + j + 1)
        plt.imshow(
          U64[random_integer, (i * n2 + j) * 30, 0, :, :], cmap=cm.viridis
        )
        plt.axis("off")
    plt.savefig("results/fig/traj_64_" + str(random_integer) + ".pdf")


def plot_gif(data: np.ndarray, fig_name: str):
  """visualize as gif"""
  fig, ax = plt.subplots(figsize=(6, 6))
  img = ax.imshow(data[0], cmap=cm.twilight)
  fig.colorbar(img)

  def update(frame):
    img.set_array(data[50 * frame])
    ax.set_title(f'Heatmap Frame: {frame}')
    return img,

  ani = FuncAnimation(fig, update, frames=100, interval=100, blit=True)
  ani.save(f"results/fig/{fig_name}.gif", writer='pillow', fps=30, dpi=100)
  plt.close()


def plot_stats(
  t_array: np.array,
  fine_traj: np.ndarray,
  baseline: np.ndarray,
  correction1: np.ndarray,
  correction2: np.ndarray,
  fig_name: str,
):

  avg_length = 500
  r = fine_traj.shape[1] // baseline.shape[1]
  plt.figure(figsize=(6, 6))
  plt.subplot(211)
  plt.plot(t_array, np.mean(fine_traj, axis=1), label="truth")
  plt.plot(
    t_array,
    np.mean(baseline, axis=1),
    label="baseline={:.3e}|MSE={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj, axis=1) - np.mean(baseline, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]) - np.mean(baseline[-avg_length:]),
      np.mean((fine_traj[:, r - 1::r] - baseline)**2)
    )
  )
  plt.plot(
    t_array,
    np.mean(correction1, axis=1),
    label="err2={:.3e}|MSE={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj, axis=1) - np.mean(correction1, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]) - np.mean(correction1[-avg_length:]),
      np.mean((fine_traj[:, r - 1::r] - correction1)**2)
    )
  )
  if correction2 is not None:
    plt.plot(
      t_array,
      np.mean(correction2, axis=1),
      label="err3={:.3e}|MSE={:.3e}".format(
        # np.linalg.norm(
        #   np.mean(fine_traj, axis=1) - np.mean(correction2, axis=1)
        # )
        np.mean(fine_traj[-avg_length:]) - np.mean(correction2[-avg_length:]),
        np.mean((fine_traj[:, r - 1::r] - correction2)**2)
      )
    )
  plt.legend()
  plt.xlabel(r"$T$")
  plt.ylabel(r"$\overline{u}$")
  plt.subplot(212)
  plt.plot(t_array, np.mean(fine_traj**2, axis=1), label="truth")
  plt.plot(
    t_array,
    np.mean(baseline**2, axis=1),
    label="baseline={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj**2, axis=1) - np.mean(baseline**2, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]**2) - np.mean(baseline[-avg_length:]**2),
    )
  )
  plt.plot(
    t_array,
    np.mean(correction1**2, axis=1),
    label="err2={:.3e}".format(
      # np.linalg.norm(
      #   np.mean(fine_traj**2, axis=1) - np.mean(correction1**2, axis=1)
      # )
      np.mean(fine_traj[-avg_length:]**2) -
      np.mean(correction1[-avg_length:]**2),
    )
  )
  if correction2 is not None:
    plt.plot(
      t_array,
      np.mean(correction2**2, axis=1),
      label="err3={:.3e}".format(
        # np.linalg.norm(
        #   np.mean(fine_traj**2, axis=1) - np.mean(correction2**2, axis=1)
        # )
        np.mean(fine_traj[-avg_length:]**2) -
        np.mean(correction2[-avg_length:]**2),
      )
    )
  plt.legend()
  plt.xlabel(r"$T$")
  plt.ylabel(r"$\overline{u^2}$")
  plt.savefig(fig_name)
  print(
    "rmse without correction: {:.3e}".format(
      np.sqrt(np.mean((baseline.T - fine_traj[:, r - 1::r].T)**2))
    )
  )
  print(
    "rmse with correction1: {:.3e}".format(
      np.sqrt(np.mean((correction1.T - fine_traj[:, r - 1::r].T)**2))
    )
  )
  if correction2 is not None:
    print(
      "rmse with correction2: {:.3e}".format(
        np.sqrt(np.mean((correction2.T - fine_traj[:, r - 1::r].T)**2))
      )
    )


def plot_stats_aux(
  t_array: np.array,
  data_list: list,
  title_list: list,
  fig_name: str,
):

  if len(data_list) != len(title_list):
    breakpoint()
    raise ValueError("The length of data_list and title_list must be the same.")
  avg_length = 1000
  plt.figure(figsize=(6, 6))
  plt.subplot(211)
  truth = data_list[0]
  plt.plot(t_array, np.mean(truth, axis=1), label=title_list[0])
  for i in range(1, len(data_list)):
    plt.plot(
      t_array,
      np.mean(data_list[i], axis=1),
      label="{}: stat_err={:.3e}".format(
        title_list[i],
        np.mean(truth[-avg_length:]) - np.mean(data_list[i][-avg_length:]),
      )
    )
  plt.legend()
  plt.xlabel(r"$T$")
  plt.ylabel(r"$\overline{u}$")
  plt.subplot(212)
  plt.plot(t_array, np.mean(truth**2, axis=1), label=title_list[0])
  for i in range(1, len(data_list)):
    plt.plot(
      t_array,
      np.mean(data_list[i]**2, axis=1),
      label="{}: stat_err={:.3e}".format(
        title_list[i],
        np.mean(truth[-avg_length:]**2) -
        np.mean(data_list[i][-avg_length:]**2),
      )
    )
  plt.legend()
  plt.xlabel(r"$T$")
  plt.ylabel(r"$\overline{u^2}$")
  plt.savefig(fig_name)
  plt.close()


def plot_error_cloudmap(
  err: np.ndarray,
  u: np.ndarray,
  u_x: np.ndarray,
  u_xx: np.ndarray,
  u_xxxx: np.ndarray,
  name: str,
):

  plt.subplot(511)
  plt.imshow(u)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$\Delta \tau$")
  plt.subplot(512)
  plt.imshow(err)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u$")
  plt.subplot(513)
  plt.imshow(u_x)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u_x$")
  plt.subplot(514)
  plt.imshow(u_xx)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u_{xx}$")
  plt.subplot(515)
  plt.imshow(u_xxxx)
  plt.colorbar()
  plt.axis("off")
  plt.ylabel(r"$u_{xxxx}$")
  plt.savefig(f"results/fig/{name}_err_dist.pdf")
  plt.close()
