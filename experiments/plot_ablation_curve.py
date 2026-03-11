import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_hex
from matplotlib import colormaps
import os
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
OUTPUT_DIR = "../figures/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
ALPHA = 1 - np.exp(-1)


def plot_segmented_line(ax, x, y, mask, style_true, style_false, label_true, label_false, **kwargs):
    """Plots a line where segments change style based on a boolean mask."""
    x, y, mask = np.asarray(x), np.asarray(y), np.asarray(mask)
    ax.plot([], [], style_true, label=label_true, **kwargs)
    ax.plot([], [], style_false, label=label_false, **kwargs)
    for i in range(len(x) - 1):
        style = style_true if mask[i] else style_false
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], style, **kwargs)


def plot_colormap_line(ax, x, y, color_values, cmap='plasma', label=None, linewidth=2):
    """Plots a line with color varying according to a colormap."""
    x, y, color_values = np.asarray(x), np.asarray(y), np.asarray(color_values)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    lc = LineCollection(
        segments, cmap=colormaps[cmap], norm=norm, linewidth=linewidth)
    lc.set_array(color_values[:-1])
    ax.add_collection(lc)
    if label:
        ax.plot([], [], color=to_hex(colormaps[cmap](0.5)),
                linewidth=linewidth, label=label)
    return lc


def pmg_af1(x):
    return x * ALPHA / 2


def pmg_af2(x, t_min=1.05, t_max=3.0):
    t_vals = np.linspace(t_min, t_max, 1000)
    y_vals = np.zeros((len(t_vals), len(x)))
    mask = np.zeros((len(t_vals), len(x)), dtype=bool)
    for i, t in enumerate(t_vals):
        term1 = (x * (t - 1)) / (t + x * (t - 1) - x)
        term2 = (x**2 / t) * (1 - 1 / np.exp(0.5))
        mask[i] = term1 < term2
        y_vals[i] = np.minimum(term1, term2)
    idx = np.argmax(y_vals, axis=0)
    return y_vals[idx, np.arange(len(x))], mask[idx, np.arange(len(x))], t_vals[idx]


def tseg_af2(x, t_min=1.05, t_max=2.0):
    t_vals = np.linspace(t_min, t_max, 1000)
    y_vals = np.zeros((len(t_vals), len(x)))
    l_thresh = (25 * ALPHA**2 - 4 * ALPHA + 4) / (24 * ALPHA**2)
    for i, t in enumerate(t_vals):
        m_prime = x + x * (x - 1) / (t - x)
        h_mid = np.maximum(0, ALPHA * m_prime)
        h_out = np.maximum(0, ALPHA * m_prime + (x / t -
                           1.5 * ALPHA + ALPHA * x * (1 - m_prime) / t))
        if t <= l_thresh:
            common = t / (4 * (ALPHA * t + 1))
            disc = np.sqrt(-24 * ALPHA**2 * t + 25 * ALPHA**2 - 4 * ALPHA + 4)
            m1, m2 = common * (5 * ALPHA + 2 - disc), common * \
                (5 * ALPHA + 2 + disc)
            y_vals[i] = np.where((x >= m1) & (x <= m2), h_mid, h_out)
        else:
            y_vals[i] = h_out
    idx = np.argmax(y_vals, axis=0)
    opt_t = t_vals[idx]
    final_mask = np.zeros(len(x), dtype=bool)
    for i, t in enumerate(opt_t):
        if t <= l_thresh:
            common = t / (4 * (ALPHA * t + 1))
            disc = np.sqrt(-24 * ALPHA**2 * t + 25 * ALPHA**2 - 4 * ALPHA + 4)
            final_mask[i] = (x[i] >= common * (5 * ALPHA + 2 - disc)
                             ) and (x[i] <= common * (5 * ALPHA + 2 + disc))
        else:
            final_mask[i] = False
    return y_vals[idx, np.arange(len(x))], final_mask, opt_t


def osepgm_af2(x, t_min=1.05, t_max=1.6):
    t_vals = np.linspace(t_min, t_max, 1000)
    y_vals = np.zeros((len(t_vals), len(x)))
    mask = np.zeros((len(t_vals), len(x)), dtype=bool)
    for i, t in enumerate(t_vals):
        m_p = x + x * (x - 1) / (t - x)
        term1, term2 = m_p / 2 + x * (2 - m_p) / (8 * t), m_p * ALPHA
        mask[i] = term1 < term2
        y_vals[i] = np.minimum(term1, term2)
    idx = np.argmax(y_vals, axis=0)
    return y_vals[idx, np.arange(len(x))], mask[idx, np.arange(len(x))], t_vals[idx]


def sg_af2(x):
    m_p = x
    h4_mid = (x + 1) / 6
    t4 = np.sqrt((x - 2) * (x - 1)) + x
    h4_out = - (t4 - 2) * (t4 - 1) / ((t4 - x) + 1e-6)
    return np.where((x >= 0.2) & (x <= 1.0), h4_mid, h4_out)


def generate_plots():
    x_dense = np.linspace(0, 1, 100)
    x_sparse = np.linspace(0, 1, 10)
    print("Generating PMG Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    y1 = pmg_af1(x_sparse)
    y2, mask, l_opt = pmg_af2(x_sparse)
    axes[0].plot(x_sparse, y1, '-', label=r'$m \alpha / 2$')
    plot_segmented_line(axes[0], x_sparse, y2, mask, 'r-', 'g--',
                        r'$\max_{\lambda} \frac{m(\lambda-1)}{\lambda+m(\lambda-1)-m}$',
                        r'$\max_{\lambda} \frac{m^2}{\lambda}(1-e^{-0.5})$')
    axes[1].plot(x_sparse, y1, '-', label=r'$m \alpha / 2$')
    lc = plot_colormap_line(
        axes[1], x_sparse, y2, l_opt, label=r'$\max_{\lambda} h_1^\lambda(m)$')
    for ax in axes:
        ax.grid(True)
        ax.set_xlabel('Monotonicity Ratio')
        ax.legend()
    axes[0].set_ylabel('Approximation Ratio')
    fig.colorbar(lc, ax=axes[1], label=r'Optimal $\lambda$')
    plt.savefig(os.path.join(OUTPUT_DIR, "png_analysis.pdf"))
    print("Generating TSEG Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    y1_p, y2_p = pmg_af1(x_dense), pmg_af2(x_dense)[0]
    y_base = np.maximum(y1_p, y2_p)
    y3, mask, l_opt = tseg_af2(x_dense)
    for ax in axes:
        ax.plot(x_dense, y_base, '-', label='PMG Max')
    plot_segmented_line(axes[0], x_dense, y3, mask, 'r-', 'g--',
                        r'Case: $\alpha m^\prime$', r'Case: $\alpha m^\prime + t_2$')
    lc = plot_colormap_line(
        axes[1], x_dense, y3, l_opt, label=r'$\max_{\lambda} h_2^\lambda(m)$')
    for ax in axes:
        ax.grid(True)
        ax.set_xlabel('Monotonicity Ratio')
        ax.legend()
    axes[0].set_ylabel('Approximation Ratio')
    fig.colorbar(lc, ax=axes[1], label=r'Optimal $\lambda$')
    plt.savefig(os.path.join(OUTPUT_DIR, "tsepg_analysis.pdf"))
    print("Generating OSEPGM Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    y_af1 = x_sparse / 2
    y_af2, mask, l_opt = osepgm_af2(x_sparse)
    for ax in axes:
        ax.plot(x_sparse, y_af1, '-', label=r'$m/2$')
    plot_segmented_line(axes[0], x_sparse, y_af2, mask,
                        'r-', 'g--', r'Term 1', r'Term 2')
    lc = plot_colormap_line(
        axes[1], x_sparse, y_af2, l_opt, label=r'$\max_{\lambda} h_3^\lambda(m)$')
    for ax in axes:
        ax.grid(True)
        ax.set_xlabel('Monotonicity Ratio')
        ax.legend()
    axes[0].set_ylabel('Approximation Ratio')
    fig.colorbar(lc, ax=axes[1], label=r'Optimal $\lambda$')
    plt.savefig(os.path.join(OUTPUT_DIR, "osepgm_analysis.pdf"))
    print("Generating Sample Greedy Analysis...")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
    y_pmg = pmg_af2(x_dense)[0]
    y_sg = sg_af2(x_dense)
    ax.plot(x_dense, y_pmg, '-', label='PMG Max')
    ax.plot(x_dense, y_sg, 'g-', label=r'$h_4(m)$')
    ax.grid(True)
    ax.set_xlabel('Monotonicity Ratio')
    ax.set_ylabel('Approximation Ratio')
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "sg_analysis.pdf"))
    print(f"Done! All figures saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_plots()
