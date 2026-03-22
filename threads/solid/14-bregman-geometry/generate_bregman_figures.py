#!/usr/bin/env python3
"""Generate Bregman geometry figures.

Produces effective rank, cosine diagnostic, and trace plots from the
2x2 factorial design (stream separation x per-layer supervision).

Output: figures/fig9_effective_rank.{pdf,png}
        figures/fig10_cosine_diagnostic.{pdf,png}
        figures/fig11_trace.{pdf,png}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS = Path("figures")
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

layers = [0, 1, 2, 3, 4, 5]
layer_labels = ['0', '1', '2', '3', '4', '5/final']

# --- Data (all 50K vocab, deconfounded) ---

erank = {
    'CASCADE + aux. loss':    [184.38, 75.45, 26.03, 36.22, 87.61, 36.76],
    'CASCADE control':        [7.61, 156.22, 8.74, 6.03, 8.41, 3.84],
    'Single-stream + aux. loss': [122.48, 48.66, 141.77, 183.12, 71.78, 17.05],
    'Single-stream control':  [8.66, 13.45, 9.23, 8.18, 8.21, 7.26],
}

trace = {
    'CASCADE + aux. loss':    [0.5056, 0.5825, 0.5677, 0.5927, 0.5660, 0.6044],
    'CASCADE control':        [0.1215, 0.5437, 0.8682, 0.5677, 0.7665, 0.5178],
    'Single-stream + aux. loss': [0.5273, 0.5459, 0.4557, 0.4440, 0.2975, 0.5682],
    'Single-stream control':  [0.0307, 0.5351, 0.3195, 0.0911, 0.0646, 0.1845],
}

cosine = {
    'CASCADE + aux. loss':    [0.2074, 0.1375, 0.2919, 0.3310, 0.4907, 0.4874],
    'CASCADE control':        [0.6106, 0.4444, 0.3723, 0.1965, 0.3253, 0.5372],
    'Single-stream + aux. loss': [0.1782, 0.1624, 0.1649, 0.2832, 0.4045, 0.5659],
    'Single-stream control':  [0.0938, 0.0673, 0.0863, 0.0963, 0.3041, 0.5176],
}

kl_adv = {
    'CASCADE + aux. loss':    [1.293, 1.505, -2.168, -0.558, -1.755, -0.765],
    'CASCADE control':        [-0.594, -0.045, 0.043, 0.152, 0.107, -0.308],
    'Single-stream + aux. loss': [0.068, 0.531, -0.240, 0.159, -0.997, -0.469],
    'Single-stream control':  [5.053, 3.190, 4.335, 0.585, -0.067, -0.149],
}

colors = {
    'CASCADE + aux. loss':       '#1f77b4',
    'CASCADE control':           '#aec7e8',
    'Single-stream + aux. loss': '#d62728',
    'Single-stream control':     '#ff9896',
}

markers = {
    'CASCADE + aux. loss':       'o',
    'CASCADE control':           's',
    'Single-stream + aux. loss': '^',
    'Single-stream control':     'D',
}


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    print(f"  saved {name}.pdf / .png")


# ========================================
# Figure 9: Effective rank across layers
# ========================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))

for name, vals in erank.items():
    ax.plot(layers, vals, marker=markers[name], color=colors[name],
            label=name, linewidth=1.5, markersize=5)

ax.set_xlabel('Layer')
ax.set_ylabel(r'Effective rank of $H^{(\ell)}$')
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels)
ax.set_ylim(0, 210)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title('Effective rank across layers')

_save(fig, "fig9_effective_rank")
plt.close(fig)


# ========================================
# Figure 10: Cosine diagnostic vs KL advantage
# ========================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))

for name in cosine:
    cos_vals = cosine[name]
    kl_vals = kl_adv[name]
    ax.scatter(cos_vals, kl_vals, color=colors[name], marker=markers[name],
               label=name, s=40, zorder=3)
    for i, (cx, ky) in enumerate(zip(cos_vals, kl_vals)):
        ax.annotate(str(i), (cx, ky), textcoords='offset points',
                    xytext=(4, 4), fontsize=7, color=colors[name])

ax.axvline(x=0.3, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.text(0.305, ax.get_ylim()[1] * 0.9, r'$\cos = 0.3$', fontsize=8, color='gray')

ax.set_xlabel(r'$\cos(\mathrm{primal}, \mathrm{dual})$')
ax.set_ylabel('KL advantage (dual over Euclidean)')
ax.legend(bbox_to_anchor=(0.02, -0.22), loc='upper left', framealpha=0.9,
          fontsize=8, ncol=2)
ax.set_title('Cosine diagnostic predicts steering method advantage')

_save(fig, "fig10_cosine_diagnostic")
plt.close(fig)


# ========================================
# Figure 11: Trace across layers
# ========================================
fig, ax = plt.subplots(figsize=(5.5, 3.5))

for name, vals in trace.items():
    ax.plot(layers, vals, marker=markers[name], color=colors[name],
            label=name, linewidth=1.5, markersize=5)

ax.set_xlabel('Layer')
ax.set_ylabel(r'$\mathrm{tr}(H^{(\ell)})$')
ax.set_xticks(layers)
ax.set_xticklabels(layer_labels)
ax.set_ylim(0, 1.0)
ax.legend(bbox_to_anchor=(0.02, -0.22), loc='upper left', framealpha=0.9,
          fontsize=9, ncol=2)
ax.set_title('Trace of Hessian (distribution concentration)')

_save(fig, "fig11_trace")
plt.close(fig)

print("All figures saved to", FIGS)
