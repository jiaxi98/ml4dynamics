import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker

sns.set_style("ticks")
sns.set_palette("pastel")

seeds = ["100", "200", "300", "400"]
ns = [512, 768, 1024]
Js = {"256": 2, "512": 2, "768": 5, "1024": 8}
losses = [
    # "first moment", "second moment", "third moment",
    "first moment traj", "second moment traj", "third moment traj"
]


dict_ = {"1/dx": [], "optimal stencil": [], "bandwidth of J": []}
for n in ns:
  optimal_stencils = []
  for seed in seeds:
    results = pickle.load(open(f"results/data/database.pkl", "rb"))
    key = "ks" + "Dirichlet-Neumann" + str(n) + seed
    data = results[key]

    order = list(dict.fromkeys(data["method"]))
    nums = [int(x[1:]) for x in order if x.startswith("s")]
    for loss in losses:
      l_min = np.nanmin(data[loss])
      weights = 0
      optimal_stencil = 0
      for _ in range(len(data["method"])):
        if data["method"][_] != "baseline":
          stencil = int(data["method"][_][1:])
          if not np.isnan(data[loss][_]):
            weight = np.exp((data[loss][_] - l_min) / np.abs(l_min))
            weights += weight
            optimal_stencil += weight * stencil
      optimal_stencils.append(optimal_stencil / weights)
      # print(optimal_stencil, weights, optimal_stencil / weights)
  dict_["1/dx"] += [n/512] * len(optimal_stencils)
  dict_["bandwidth of J"] += [Js[str(n)]] * len(optimal_stencils)
  dict_["optimal stencil"] += optimal_stencils

fig, ax = plt.subplots(figsize=(6, 5))
# sns.violinplot(data=dict_, x="1/dx", y="optimal stencil")

# xs = np.array(dict_["1/dx"])
xs = np.array(dict_["bandwidth of J"])
coeff = np.concatenate((xs[:, None], np.ones(len(xs))[:, None]), axis=1)
coeff = np.linalg.lstsq(coeff, np.array(dict_["optimal stencil"])[:, None], rcond=None)[0]
plt.plot(xs, dict_["optimal stencil"], "o")
plt.plot(xs, coeff[0] * xs + coeff[1], color="red")
plt.savefig("results/fig/violin_jac.pdf", dpi=300)
