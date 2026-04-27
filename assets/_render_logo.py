"""
Render the neptune logo at 2× resolution as a PNG.

Run:  python assets/_render_logo.py

This produces assets/logo.png (banner) and assets/logo_square.png (square icon).
The canonical source remains assets/logo.svg.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TEAL_LIGHT = "#5ddef4"
TEAL = "#26c6da"
TEAL_DARK = "#00838f"
ACCENT = "#80deea"
INK = "#1a8c9c"
WHITE = "#ffffff"

cmap = LinearSegmentedColormap.from_list(
    "trident", [TEAL_LIGHT, TEAL, TEAL_DARK]
)


def _gradient_segments(xs, ys, n_seg=80, lw=14.0):
    """Return a LineCollection that draws a gradient along the path."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # interpolate to n_seg points along the polyline by arc length
    s = np.r_[0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    s_norm = s / s[-1]
    s_new = np.linspace(0, 1, n_seg)
    x_new = np.interp(s_new, s_norm, xs)
    y_new = np.interp(s_new, s_norm, ys)
    pts = np.column_stack([x_new, y_new]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # color along the *vertical* axis (y), not arc length, so all elements share
    # a single global gradient
    y_mid = (segs[:, 0, 1] + segs[:, 1, 1]) / 2.0
    return segs, y_mid


def render(banner=True, dpi=200):
    if banner:
        fig_w, fig_h = 7.6, 2.2     # inches
        figsize = (fig_w, fig_h)
        outfile = os.path.join(HERE, "logo.png")
    else:
        figsize = (4.0, 4.0)
        outfile = os.path.join(HERE, "logo_square.png")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 760 if banner else 220)
    ax.set_ylim(220, 0)              # flip y so SVG-style coords work
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Trident origin — same coords as the SVG
    Tx, Ty = 40, 14

    paths = []
    # Center prong (straight)
    paths.append(([Tx + 90, Tx + 90], [Ty + 6, Ty + 120]))
    # Approximate quadratic Bezier for "M 32 120 V 38 Q 32 16 14 16"
    bez_t = np.linspace(0, 1, 30)
    qbx = (1 - bez_t) ** 2 * 32 + 2 * (1 - bez_t) * bez_t * 32 + bez_t ** 2 * 14
    qby = (1 - bez_t) ** 2 * 38 + 2 * (1 - bez_t) * bez_t * 16 + bez_t ** 2 * 16
    left_x = np.r_[32, qbx] + Tx
    left_y = np.r_[120, qby] + Ty
    # downsample first vertical segment
    left_x = np.r_[Tx + 32, left_x]
    left_y = np.r_[Ty + 120, left_y]
    paths.append((left_x, left_y))
    # Right prong (mirror)
    qbx_r = (1 - bez_t) ** 2 * 148 + 2 * (1 - bez_t) * bez_t * 148 + bez_t ** 2 * 166
    qby_r = (1 - bez_t) ** 2 * 38 + 2 * (1 - bez_t) * bez_t * 16 + bez_t ** 2 * 16
    right_x = np.r_[148, qbx_r] + Tx
    right_y = np.r_[120, qby_r] + Ty
    paths.append((right_x, right_y))
    # Crossbar
    paths.append(([Tx + 22, Tx + 158], [Ty + 120, Ty + 120]))
    # Handle
    paths.append(([Tx + 90, Tx + 90], [Ty + 120, Ty + 186]))

    # Build a global gradient from y∈[Ty, Ty+186]
    y_min, y_max = Ty + 0, Ty + 186
    all_segs = []
    all_ymid = []
    for xs, ys in paths:
        segs, ymid = _gradient_segments(xs, ys, n_seg=120)
        all_segs.append(segs)
        all_ymid.append(ymid)
    segs = np.concatenate(all_segs)
    ymids = np.concatenate(all_ymid)
    norm = (ymids - y_min) / (y_max - y_min)
    colors = cmap(np.clip(norm, 0, 1))
    lc = LineCollection(
        segs, colors=colors, linewidths=14, capstyle="round", joinstyle="round",
        antialiased=True,
    )
    ax.add_collection(lc)

    # Particle "tips" at the prong points
    tips = [(Tx + 14, Ty + 16), (Tx + 90, Ty + 6), (Tx + 166, Ty + 16)]
    for tx, ty in tips:
        # Outer glow
        for r, a in [(13, 0.35), (9, 0.55), (6, 0.85)]:
            ax.add_patch(Circle((tx, ty), r, color=ACCENT, alpha=a, ec="none"))
        ax.add_patch(Circle((tx, ty), 3.5, color=WHITE, ec="none", zorder=5))

    # Subtle vertex dot at the trident base
    ax.add_patch(Circle((Tx + 90, Ty + 120), 4.5, color=TEAL, alpha=0.75, ec="none"))

    if banner:
        # Wordmark
        ax.text(
            240, 122, "neptune",
            fontfamily="sans-serif", fontsize=58, fontweight="bold",
            color=INK, va="baseline",
        )
        # Tagline (extra spaces fake letter-spacing in matplotlib)
        ax.text(
            243, 158, "N E U T R I N O   ·   T R I D E N T   ·   E V E N T   ·   G E N E R A T O R",
            fontfamily="sans-serif", fontsize=8, fontweight="medium",
            color=TEAL, va="baseline",
        )

    plt.savefig(outfile, dpi=dpi, transparent=True,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {outfile} ({os.path.getsize(outfile):,} bytes)")


if __name__ == "__main__":
    render(banner=True)
    render(banner=False)
