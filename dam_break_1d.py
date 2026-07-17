"""
1D Dam-Break Simulation — Saint-Venant (Shallow Water) Equations
==================================================================

Solves the 1D shallow water equations:

    dh/dt + d(hu)/dx = 0
    d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx = 0

using a finite-volume method with an HLL approximate Riemann solver
and explicit Euler time stepping (CFL-limited).

Also computes the exact Stoker (1957) analytical solution for a
dam-break over a wet bed, so the numerical result can be validated.

Outputs:
    dam_break_comparison.png  -> numerical vs analytical at one time
    dam_break_snapshots.png   -> numerical profiles at several times
    dam_break_animation.gif   -> animated depth/velocity evolution

Author: generated with Claude
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

g = 9.81  # gravitational acceleration [m/s^2]

# Visual styling for the channel / dam structure
BED_COLOR = "#9c8362"
WATER_COLOR = "#2e86c1"
DAM_COLOR = "#5a5a5a"
DAM_EDGE = "#3a3a3a"


# ----------------------------------------------------------------------
# Numerical solver: finite volume + HLL flux
# ----------------------------------------------------------------------

def hll_flux(hL, uL, hR, uR):
    """HLL approximate Riemann solver flux for shallow water equations."""
    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)

    # wave speed estimates
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    FL = np.array([hL * uL, hL * uL**2 + 0.5 * g * hL**2])
    FR = np.array([hR * uR, hR * uR**2 + 0.5 * g * hR**2])
    UL = np.array([hL, hL * uL])
    UR = np.array([hR, hR * uR])

    if SL >= 0:
        return FL
    elif SR <= 0:
        return FR
    else:
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)


def simulate_dam_break(L=50.0, nx=400, x0=25.0, hL=2.0, hR=0.5,
                        t_end=4.0, cfl=0.4, save_times=None, dry_tol=1e-6):
    """
    Run a finite-volume dam-break simulation.

    Parameters
    ----------
    L        : channel length [m]
    nx       : number of cells
    x0       : dam location [m]
    hL, hR   : upstream / downstream initial depths [m]
    t_end    : simulation end time [s]
    cfl      : Courant number (<=1 for stability)
    save_times : list of times [s] at which to store a snapshot

    Returns
    -------
    x        : cell-center coordinates
    snapshots: dict {t: (h, u)} of saved snapshots
    """
    dx = L / nx
    x = (np.arange(nx) + 0.5) * dx

    h = np.where(x < x0, hL, hR).astype(float)
    u = np.zeros(nx)
    h = np.maximum(h, dry_tol)

    if save_times is None:
        save_times = np.linspace(0, t_end, 6)
    save_times = sorted(save_times)
    snapshots = {}

    t = 0.0
    next_save_idx = 0
    if abs(t - save_times[0]) < 1e-9:
        snapshots[save_times[0]] = (h.copy(), u.copy())
        next_save_idx = 1

    while t < t_end:
        c = np.sqrt(g * h)
        max_speed = np.max(np.abs(u) + c)
        dt = cfl * dx / max_speed
        if t + dt > t_end:
            dt = t_end - t

        # ghost cells: transmissive (zero-gradient) boundaries
        h_ext = np.concatenate(([h[0]], h, [h[-1]]))
        u_ext = np.concatenate(([u[0]], u, [u[-1]]))

        fluxes = np.zeros((nx + 1, 2))
        for i in range(nx + 1):
            fluxes[i] = hll_flux(h_ext[i], u_ext[i], h_ext[i + 1], u_ext[i + 1])

        hu = h * u
        h_new = h - dt / dx * (fluxes[1:, 0] - fluxes[:-1, 0])
        hu_new = hu - dt / dx * (fluxes[1:, 1] - fluxes[:-1, 1])

        h_new = np.maximum(h_new, dry_tol)
        u_new = hu_new / h_new

        h, u = h_new, u_new
        t += dt

        while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 1e-9:
            snapshots[save_times[next_save_idx]] = (h.copy(), u.copy())
            next_save_idx += 1

    return x, snapshots


# ----------------------------------------------------------------------
# Analytical solution: Stoker (1957) dam break over a wet bed
# ----------------------------------------------------------------------

def _solve_hm(hL, hR, tol=1e-12, maxiter=100):
    """Newton solve for the intermediate depth h_m in the Stoker solution."""
    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)

    def f(hm):
        cm = np.sqrt(g * hm)
        return 2 * (cL - cm) - (hm - hR) * np.sqrt(g * (hm + hR) / (2 * hR * hm))

    def fprime(hm, eps=1e-8):
        return (f(hm + eps) - f(hm - eps)) / (2 * eps)

    hm = 0.5 * (hL + hR)  # initial guess
    for _ in range(maxiter):
        fx = f(hm)
        dfx = fprime(hm)
        hm_new = hm - fx / dfx
        if hm_new <= 0:
            hm_new = hm / 2
        if abs(hm_new - hm) < tol:
            hm = hm_new
            break
        hm = hm_new
    return hm


def stoker_solution(x, t, x0, hL, hR):
    """
    Exact Stoker solution for a dam break over a wet bed (hR > 0),
    evaluated at positions x and time t.

    Returns h(x), u(x).
    """
    if t <= 0:
        h = np.where(x < x0, hL, hR).astype(float)
        u = np.zeros_like(x)
        return h, u

    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)
    hm = _solve_hm(hL, hR)
    cm = np.sqrt(g * hm)
    um = 2 * (cL - cm)

    # shock (bore) speed, from mass/momentum jump conditions
    S = um * hm / (hm - hR) if hm != hR else cR

    xi = (x - x0) / t  # self-similar variable

    h = np.zeros_like(x)
    u = np.zeros_like(x)

    for i, s in enumerate(xi):
        if s <= -cL:
            h[i], u[i] = hL, 0.0
        elif s <= um - cm:
            # rarefaction fan
            u[i] = 2.0 / 3.0 * (cL + s)
            c_loc = 1.0 / 3.0 * (2 * cL - s)
            h[i] = c_loc**2 / g
        elif s <= S:
            h[i], u[i] = hm, um
        else:
            h[i], u[i] = hR, 0.0

    return h, u


# ----------------------------------------------------------------------
# Channel / dam structure drawing helpers
# ----------------------------------------------------------------------

def draw_channel_bed(ax, x_min, x_max, bed_depth):
    """Draw a solid channel bed below y=0."""
    ax.add_patch(Rectangle((x_min, -bed_depth), x_max - x_min, bed_depth,
                            color=BED_COLOR, zorder=1, lw=0))


def draw_dam(ax, x0, wall_height, frac=1.0, thickness=0.8, gap=2.2,
             stub_height_frac=0.22, zorder=6):
    """
    Draw the dam structure at x0, at collapse progress `frac`:

        frac = 0.0  -> fully intact, single solid wall blocking the channel
        frac = 1.0  -> fully broken, two short pier stubs with an open gap

    Values in between smoothly erode the middle section down to nothing
    while the two end piers shrink from full wall height to stub height,
    giving a continuous collapse animation instead of a hard cut.
    """
    frac = np.clip(frac, 0.0, 1.0)
    stub_h = wall_height * stub_height_frac

    left_x = x0 - gap / 2 - thickness
    right_x = x0 + gap / 2
    pier_h = wall_height - frac * (wall_height - stub_h)
    mid_h = wall_height * (1.0 - frac)

    patches = []
    style = dict(facecolor=DAM_COLOR, edgecolor=DAM_EDGE, lw=1.2, zorder=zorder)
    if frac > 0.02:
        style["hatch"] = "///"

    for px in (left_x, right_x):
        p = Rectangle((px, 0), thickness, pier_h, **style)
        ax.add_patch(p)
        patches.append(p)

    if mid_h > 1e-3:
        p = Rectangle((x0 - gap / 2, 0), gap, mid_h, **style)
        ax.add_patch(p)
        patches.append(p)

    return patches


def draw_water(ax, x, h, zorder=3):
    """Fill the water surface down to the channel bed (y=0)."""
    return ax.fill_between(x, 0, h, color=WATER_COLOR, alpha=0.55, zorder=zorder)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def make_comparison_plot(L, x0, hL, hR, t_end, t_compare, filename):
    x, snaps = simulate_dam_break(L=L, nx=500, x0=x0, hL=hL, hR=hR,
                                   t_end=t_end, save_times=[0, t_compare])
    h_num, u_num = snaps[t_compare]
    h_an, u_an = stoker_solution(x, t_compare, x0, hL, hR)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    bed_depth = 0.15 * hL
    draw_channel_bed(axes[0], x.min(), x.max(), bed_depth)
    draw_water(axes[0], x, h_num)
    draw_dam(axes[0], x0, wall_height=hL * 1.3, frac=1.0)

    axes[0].plot(x, h_num, color="#0b4f7a", lw=2, label="Numerical (HLL)", zorder=4)
    axes[0].plot(x, h_an, "--", color="#d62728", lw=2, label="Analytical (Stoker)", zorder=4)
    axes[0].set_ylim(-bed_depth, hL * 1.35)
    axes[0].set_ylabel("Water depth h [m]")
    axes[0].set_title(f"1D Dam Break — comparison at t = {t_compare:.2f} s\n"
                       f"(gray = broken dam piers, gap = breach)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, u_num, color="#1f77b4", lw=2, label="Numerical (HLL)")
    axes[1].plot(x, u_an, "--", color="#d62728", lw=2, label="Analytical (Stoker)")
    axes[1].set_ylabel("Velocity u [m/s]")
    axes[1].set_xlabel("x [m]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_snapshot_plot(L, x0, hL, hR, t_end, times, filename):
    x, snaps = simulate_dam_break(L=L, nx=500, x0=x0, hL=hL, hR=hR,
                                   t_end=t_end, save_times=times)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bed_depth = 0.15 * hL
    draw_channel_bed(ax, x.min(), x.max(), bed_depth)
    draw_dam(ax, x0, wall_height=hL * 1.3, frac=1.0)

    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(times)))
    for c, t in zip(cmap, times):
        h, u = snaps[t]
        ax.plot(x, h, color=c, lw=2, label=f"t = {t:.2f} s", zorder=4)

    ax.set_ylim(-bed_depth, hL * 1.35)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Water depth h [m]")
    ax.set_title("1D Dam Break — depth profile evolution\n"
                 "(gray piers mark the breached dam)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def make_animation(L, x0, hL, hR, t_end, filename, n_frames=80,
                    pre_break_frames=6):
    """
    Animate the dam break. The first `pre_break_frames` frames show the
    dam wall fully intact and holding back still water (t=0), then the
    wall breaks open and the flood wave propagates for the rest of the
    animation.
    """
    times = np.linspace(0, t_end, n_frames)
def make_animation(L, x0, hL, hR, t_end, filename, n_frames=80,
                    pre_break_frames=6, collapse_frames=10):
    """
    Animate the dam break in three phases:

      1. Hold  - dam sits fully intact, still water on both sides (t=0)
      2. Collapse - the wall erodes smoothly from solid block to two
         broken piers over `collapse_frames` frames (water not yet
         moving - the structural failure is fast relative to the flow)
      3. Flow - the piers stay put as the flood wave propagates,
         stepping through the simulated times up to t_end
    """
    times = np.linspace(0, t_end, n_frames)
    x, snaps = simulate_dam_break(L=L, nx=300, x0=x0, hL=hL, hR=hR,
                                   t_end=t_end, save_times=times)

    bed_depth = 0.15 * hL
    wall_height = hL * 1.3
    h0, u0 = snaps[times[0]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    draw_channel_bed(ax1, x.min(), x.max(), bed_depth)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(-bed_depth, wall_height * 1.05)
    ax1.set_ylabel("Depth h [m]")
    ax1.grid(alpha=0.3)
    title = ax1.set_title("t = 0.00 s  (dam intact)")

    line_u, = ax2.plot(x, u0, color="#d62728", lw=2)
    u_all = np.concatenate([snaps[t][1] for t in times])
    ax2.set_ylim(u_all.min() - 0.2, u_all.max() + 0.2)
    ax2.set_ylabel("Velocity u [m/s]")
    ax2.set_xlabel("x [m]")
    ax2.grid(alpha=0.3)

    fig.tight_layout()

    # mutable containers so the update function can remove old artists
    state = {"water": None, "dam_patches": []}

    def clear_dynamic_artists():
        if state["water"] is not None:
            state["water"].remove()
            state["water"] = None
        for p in state["dam_patches"]:
            p.remove()
        state["dam_patches"] = []

    def ease_in_out(s):
        """Smoothstep easing so the collapse accelerates then settles."""
        return s * s * (3 - 2 * s)

    n_hold = pre_break_frames
    n_collapse = collapse_frames
    n_flow = len(times)
    total_frames = n_hold + n_collapse + n_flow

    def update(frame):
        clear_dynamic_artists()

        if frame < n_hold:
            # Phase 1: intact, still water
            t, h, u = 0.0, h0, u0
            frac = 0.0
            status = "dam intact"
        elif frame < n_hold + n_collapse:
            # Phase 2: wall erodes away, water hasn't started moving yet
            t, h, u = 0.0, h0, u0
            s = (frame - n_hold + 1) / n_collapse
            frac = ease_in_out(s)
            status = "breaking..."
        else:
            # Phase 3: piers fixed open, flood wave propagates
            t_idx = min(frame - n_hold - n_collapse, n_flow - 1)
            t = times[t_idx]
            h, u = snaps[t]
            frac = 1.0
            status = "breach open"

        water = draw_water(ax1, x, h)
        state["water"] = water
        state["dam_patches"] = draw_dam(ax1, x0, wall_height=wall_height, frac=frac)

        title.set_text(f"t = {t:.2f} s  ({status})")
        line_u.set_ydata(u)

        return [water, line_u, title] + state["dam_patches"]

    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                    interval=60, blit=False)
    anim.save(filename, writer="pillow", fps=15)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Problem setup: channel with a dam at x0, hL upstream, hR downstream (wet bed)
    L, x0 = 50.0, 25.0
    hL, hR = 2.0, 0.5
    t_end = 4.0

    print("Running numerical vs analytical comparison...")
    make_comparison_plot(L, x0, hL, hR, t_end, t_compare=2.5,
                          filename="dam_break_comparison.png")

    print("Running multi-time snapshot plot...")
    make_snapshot_plot(L, x0, hL, hR, t_end,
                        times=[0, 0.5, 1.0, 1.5, 2.5, 4.0],
                        filename="dam_break_snapshots.png")

    print("Building animation (this takes a bit longer)...")
    make_animation(L, x0, hL, hR, t_end, filename="dam_break_animation.gif")

    print("Done. Files written: dam_break_comparison.png, "
          "dam_break_snapshots.png, dam_break_animation.gif")
