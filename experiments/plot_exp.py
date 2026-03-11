import numpy as np
import json
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'

algo_names_lst = [
    "Positive Modified Greedy",
    "Positive Greedy Max",
    "Two Set Enumeration Positive Greedy",
    "One Set Enumeration Positive Greedy Max",
    "Sample Greedy",
    "Inapproximability",
]
t_markers = ['o', 's', 'D', '^', 'v', '<', '+']
markers = {n: m for m, n in zip(t_markers, algo_names_lst)}
markers['Deterministic Linear Approximation'] = 'o'
markers['Randomized Linear Approximation'] = 's'
algo_name_mapping = {
    "Positive Modified Greedy": "Positive Modified Greedy",
    "Positive Greedy Max": "Positive Greedy+Max",
    "Two Set Enumeration Positive Greedy": "Two Set Enumeration Positive Greedy",
    "One Set Enumeration Positive Greedy Max": "One Set Enumeration Positive Greedy+Max",
    "Sample Greedy": "Sample Greedy",
    "Deterministic Linear Approximation": "Deterministic Linear Approximation",
    "Randomized Linear Approximation": "Randomized Linear Approximation",
}
algo_name_mask = {
    "Positive Modified Greedy": True,
    "Positive Greedy Max": True,
    "Two Set Enumeration Positive Greedy": True,
    "One Set Enumeration Positive Greedy Max": True,
    "Sample Greedy": True,
    "Deterministic Linear Approximation": True,
    "Randomized Linear Approximation": True,
}
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
algo_name_color = {
    "Positive Modified Greedy": default_colors[0],
    "Positive Greedy Max": default_colors[1],
    "Two Set Enumeration Positive Greedy": default_colors[2],
    "One Set Enumeration Positive Greedy Max": default_colors[3],
    "Sample Greedy": default_colors[4],
    "Deterministic Linear Approximation": default_colors[5],
    "Randomized Linear Approximation": default_colors[6],
}


def load_data_and_x_axis(file_path, budget_ratios):
    """
    Loads JSON and extracts both the Y-values (AF) and the 
    X-axis (monotonicity_ratio) from the data itself.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    formatted_list = []
    x_axis = None
    for br in budget_ratios:
        br_str = str(br)
        if br_str not in data:
            formatted_list.append({})
            continue
        br_data = data[br_str]
        algo_dict = {}
        for algo_name, results in br_data.items():
            algo_dict[algo_name] = [res['AF'] for res in results]
            if x_axis is None:
                x_axis = [res['monotonicity_ratio'] for res in results]
        formatted_list.append(algo_dict)
    return formatted_list, x_axis


budget_ratios = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
algo_ms_ys_dict_movie_lst, new_ms = load_data_and_x_axis(
    "result/exp_rst_movie.json", budget_ratios)
algo_ms_ys_dict_influence_lst, _ = load_data_and_x_axis(
    "result/exp_rst_influence.json", budget_ratios)
fig, axs = plt.subplots(3, 4, figsize=(16, 12), sharey=True)
for i in range(len(budget_ratios)):
    row = i % 3
    col_group = 0 if i < 3 else 1
    ax_m = axs[row, col_group]
    movie_data = algo_ms_ys_dict_movie_lst[i]
    for algo_name, ys in movie_data.items():
        if algo_name_mask[algo_name]:
            ax_m.plot(new_ms, ys[:len(new_ms)],
                      label=algo_name_mapping[algo_name],
                      marker=markers[algo_name],
                      color=algo_name_color[algo_name],
                      alpha=0.8, markersize=5)
    ax_m.set_title(
        f"Movie Rec. (Budget Ratio: {budget_ratios[i]})", fontsize=16, fontweight='bold')
    ax_i = axs[row, col_group + 2]
    ie_data = algo_ms_ys_dict_influence_lst[i]
    for algo_name, ys in ie_data.items():
        if algo_name_mask[algo_name]:
            ax_i.plot(new_ms, ys[:len(new_ms)],
                      label=algo_name_mapping[algo_name],
                      marker=markers[algo_name],
                      color=algo_name_color[algo_name],
                      alpha=0.8, markersize=5)
    ax_i.set_title(
        f"IE Marketing (Budget Ratio: {budget_ratios[i]})", fontsize=16, fontweight='bold')
for r in range(3):
    axs[r, 0].set_ylabel('Empirical Ratio', fontsize=16)
    axs[r, 2].set_ylabel('Empirical Ratio', fontsize=16)
for c in range(4):
    axs[2, c].set_xlabel('Monotonicity Ratio', fontsize=16)
for ax in axs.flat:
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=10)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=14, loc='lower center',
           bbox_to_anchor=(0.5, -0.08), ncol=len(labels) // 2, frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("../figures/empirical_approximation_ratio_sota_3x4.pdf",
            bbox_inches='tight')
