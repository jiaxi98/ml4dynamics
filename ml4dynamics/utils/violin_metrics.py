import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

if __name__ == "__main__":
    # Load aposteriori_metrics.pkl
    filename = "results/aposteriori_metrics.pkl"
    with open(filename, "rb") as f:
        metrics = pickle.load(f)

    # List of metric names to plot (sample-wise)
    metric_names = [
        "l2_list",
        "first_moment_list",
        "second_moment_list",
        "third_moment_list",
        "first_moment_traj_list",
        "second_moment_traj_list",
        "third_moment_traj_list"
    ]

    fig_dir = "results/fig"
    os.makedirs(fig_dir, exist_ok=True)

    # Group keys by r value
    r_to_keys = {}
    for key in metrics:
        match = re.match(r"[a-z]+_r(\d+)_s(\d+)", key)
        if match:
            r = int(match.group(1))
            s = int(match.group(2))
            r_to_keys.setdefault(r, []).append((s, key))

    for metric_name in metric_names:
        for r, s_key_list in r_to_keys.items():
            # Sort by stencil size
            s_key_list = sorted(s_key_list)
            plot_data = []
            for s, key in s_key_list:
                if metric_name not in metrics[key]:
                    continue
                data = np.array(metrics[key][metric_name])  # shape (num_samples, 2)
                baseline = data[:, 0]
                ours = data[:, 1]
                for val in baseline:
                    plot_data.append({"method": "baseline", "error": val, "stencil_size": s})
                for val in ours:
                    plot_data.append({"method": "ours", "error": val, "stencil_size": s})
            if plot_data:
                df = pd.DataFrame(plot_data)
                plt.figure(figsize=(10, 6))
                sns.violinplot(x="stencil_size", y="error", hue="method", data=df, split=True)
                plt.title(f"r={r} - {metric_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"r{r}_{metric_name}_violin.png"), dpi=300)
                plt.close()

    # Plot corr1 and corr2 distributions (aggregate over time, grouped by r)
    avg_length = 1000  # or set to your preferred value
    for r, s_key_list in r_to_keys.items():
        # Sort by stencil size
        s_key_list = sorted(s_key_list)
        plot_data = []
        for s, key in s_key_list:
            if "corr1" not in metrics[key] or "corr2" not in metrics[key]:
                continue
            corr1_data = np.array(metrics[key]["corr1"])  # shape (n_sample, step_num)
            corr2_data = np.array(metrics[key]["corr2"])  # shape (n_sample, step_num)
            T = corr1_data.shape[1]
            use_length = min(avg_length, T)
            corr1_agg = np.mean(corr1_data[:, -use_length:], axis=1)  # shape (n_sample,)
            corr2_agg = np.mean(corr2_data[:, -use_length:], axis=1)  # shape (n_sample,)
            for val in corr1_agg:
                plot_data.append({"method": "baseline", "correlation": val, "stencil_size": s})
            for val in corr2_agg:
                plot_data.append({"method": "ours", "correlation": val, "stencil_size": s})
        if plot_data:
            df = pd.DataFrame(plot_data)
            plt.figure(figsize=(10, 6))
            sns.violinplot(x="stencil_size", y="correlation", hue="method", data=df, split=True)
            plt.title(f"r={r} - correlation (mean over last {avg_length} steps)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"r{r}_correlation_violin.png"), dpi=300)
            plt.close()
