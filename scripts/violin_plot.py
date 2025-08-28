import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mticker

sns.set_style("ticks")
sns.set_palette("pastel")

seeds = ["100"]
stencil = ["7", "9", "11"]
losses = [
    "l2", "first moment", "second moment", "third moment",
    "first moment traj", "second moment traj", "third moment traj", "corr"
]

for seed in seeds:
    results = pickle.load(open(f"results/data/database.pkl", "rb"))
    key = "ks" + "Dirichlet-Neumann" + "512" + str(seed)
    data = results[key]

    order = list(dict.fromkeys(data["method"]))
    order = sorted(order, key=lambda x: (x != "baseline", int(x[1:]) if x.startswith("s") else float("inf")))
    for l in losses:
        methods_with_nan = []
        for m in order:
            vals = [
                float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
                for mm, v in zip(data["method"], data[l]) if mm == m
            ]
            if any(np.isnan(vals)):
                methods_with_nan.append(m)

        data_plot = {k: list(v) for k, v in data.items()}
        for m in methods_with_nan:
            for i, mm in enumerate(data_plot["method"]):
                if mm == m:
                    data_plot[l][i] = np.nan

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(
            data=data_plot,
            x="method", y=l,
            order=order, inner=None,
            density_norm="width", cut=2,
            linewidth=0.8, ax=ax
        )

        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.yaxis.set_major_locator(mticker.AutoLocator())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(
            axis="y", which="both",
            length=3.5, width=1.0, color="black", labelsize=10, direction="out"
        )
        ax.spines["left"].set_visible(True)
        ax.yaxis.set_ticks_position("left")
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        y_span = ylim[1] - ylim[0]
        violin_width = 0.8
        half_bar = violin_width * 0.25

        for i, m in enumerate(order):
            if m in methods_with_nan:
                ax.text(
                    i, 0.02, "nan",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom",
                    color="black", fontsize=10, clip_on=False
                )
                continue

            vals = [
                float(v)
                for mm, v in zip(data["method"], data[l])
                if mm == m and not (isinstance(v, float) and np.isnan(v))
            ]
            if not vals:
                continue

            mean_val = np.mean(vals)
            ax.plot(
                [xlim[0], i - half_bar],
                [mean_val, mean_val],
                color="black", linestyle="--",
                linewidth=1.2, zorder=3, clip_on=False
            )
            ax.plot(
                [i - half_bar, i + half_bar],
                [mean_val, mean_val],
                color="white", linewidth=1,
                solid_capstyle="butt", zorder=4
            )
            ax.text(
                i, mean_val + y_span * 0.01,
                f"{mean_val:.2f}",
                # transform=ax.get_yaxis_transform(),
                ha="left", va="bottom",
                color="black", fontsize=9, clip_on=False
            )

        # ax.set_title(f"seed {seed}")
        # ax.set_xlabel("method")
        # ax.set_ylabel(l)
        fig.subplots_adjust(left=0.15, right=0.95)
        fig.savefig(
            f"results/fig/violin_{l.replace(' ', '_')}_{seed}.pdf",
            dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
