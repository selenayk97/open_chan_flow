# 1D Dam-Break Simulation — README

This project solves the classic **dam-break problem** using the 1D shallow
water (Saint-Venant) equations, with a finite-volume numerical solver and
an exact analytical solution for validation.

Run it with:

```bash
python3 dam_break_1d.py
```

which produces:

- `dam_break_comparison.png` — numerical vs. analytical profiles at one time
- `dam_break_snapshots.png` — numerical depth profiles at several times
- `dam_break_animation.gif` — animated collapse + flood wave propagation

---

## 1. Governing equations

The flow is modeled with the **1D Saint-Venant (shallow water) equations**,
which describe conservation of mass and momentum for a shallow layer of
fluid on a flat, frictionless bed:

**Continuity (mass conservation):**

```
∂h/∂t + ∂(hu)/∂x = 0
```

**Momentum:**

```
∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = 0
```

where:

| Symbol | Meaning                     |
|--------|------------------------------|
| `h(x,t)` | water depth [m]             |
| `u(x,t)` | depth-averaged velocity [m/s] |
| `g`      | gravitational acceleration [9.81 m/s²] |
| `x`      | position along the channel [m] |
| `t`      | time [s]                    |

These are written in **conservative form** as:

```
∂U/∂t + ∂F(U)/∂x = 0

U = [ h  ]        F(U) = [   hu        ]
    [ hu ]                [ hu² + ½gh² ]
```

This is the form the finite-volume solver actually integrates.

The term `½gh²` is the hydrostatic pressure force integrated over depth —
it's what pushes water from the deep side toward the shallow side.

### Wave speed

Small-amplitude surface waves travel relative to the flow at the
**shallow-water wave speed**:

```
c = √(gh)
```

This shows up constantly in the solution: it sets the numerical stability
limit and appears throughout the analytical solution below.

### Froude number (for reference)

```
Fr = u / c = u / √(gh)
```

`Fr < 1` is subcritical, `Fr > 1` is supercritical — not used directly in
this code, but it's the natural dimensionless number for this system.

---

## 2. Numerical method: finite volume + HLL Riemann solver

The domain is divided into `nx` cells. Each cell stores an average value
of `U = [h, hu]`. Every timestep, the solver:

1. Computes the flux `F` at each cell interface using the **HLL
   (Harten–Lax–van Leer) approximate Riemann solver**
2. Updates each cell with the standard finite-volume update formula
3. Restricts the timestep with a CFL condition for stability

### HLL flux

At each interface, with left state `(hL, uL)` and right state `(hR, uR)`:

```
wave speed estimates:
    S_L = min(uL - cL, uR - cR)
    S_R = max(uL + cL, uR + cR)

HLL flux:
    F_HLL = F_L                                          if S_L ≥ 0
    F_HLL = F_R                                          if S_R ≤ 0
    F_HLL = (S_R·F_L - S_L·F_R + S_L·S_R·(U_R - U_L)) / (S_R - S_L)   otherwise
```

This approximates the true solution of the Riemann problem (the jump
between two constant states) by bracketing it between the fastest
left-going and right-going wave speeds. It's simple, robust, and handles
shocks (bores) and rarefactions without oscillating.

### Finite-volume update

For cell `i` with interface fluxes `F_{i-1/2}` and `F_{i+1/2}`:

```
U_i^{n+1} = U_i^n - (Δt/Δx) · (F_{i+1/2} - F_{i-1/2})
```

### CFL condition (stability)

The timestep is limited by how fast information (waves) can cross a cell:

```
Δt = CFL · Δx / max(|u| + c)
```

with `CFL ≤ 1` (the code uses `0.4` by default, for extra margin).

### Boundary conditions

Both ends of the channel use **transmissive (zero-gradient)** boundaries —
ghost cells just copy the nearest interior cell, so waves exit the domain
without reflecting back.

---

## 3. Analytical solution: Stoker's solution

For an idealized dam break over a **wet bed** (`hR > 0`, flat, frictionless
channel), there's an exact solution, due to Stoker (1957), used here to
validate the numerical scheme.

At `t = 0`, the dam at `x = x0` fails instantly, releasing depth `hL`
(upstream) into depth `hR` (downstream), both initially at rest.

The solution depends only on the **self-similar variable**:

```
ξ = (x - x0) / t
```

and splits into four regions:

```
 hL, u=0  |  rarefaction fan  |  hm, u=um  |  hR, u=0
──────────┼───────────────────┼────────────┼──────────→  x
       ξ = -cL            ξ = um - cm    ξ = S
```

**1. Undisturbed upstream** (`ξ ≤ -cL`):

```
h = hL,   u = 0
```

**2. Rarefaction fan** (`-cL ≤ ξ ≤ um - cm`):

A smooth expansion wave connects `hL` to the intermediate state:

```
u(ξ) = (2/3)(cL + ξ)
c(ξ) = (1/3)(2cL - ξ)
h(ξ) = c(ξ)² / g
```

**3. Constant intermediate state** (`um - cm ≤ ξ ≤ S`):

```
h = hm,   u = um
```

**4. Undisturbed downstream, beyond the bore** (`ξ > S`):

```
h = hR,   u = 0
```

where `cL = √(g·hL)`, `cR = √(g·hR)`, `cm = √(g·hm)`.

### Solving for the intermediate depth `hm`

`hm` and `um` are found from two physical constraints that must agree:

- **Rarefaction relation** (Riemann invariant along the fan):

```
um = 2(cL - cm)
```

- **Shock (bore) jump condition** (mass + momentum conservation across
  the moving bore):

```
um = (hm - hR) · √( g(hm + hR) / (2·hR·hm) )
```

Combining these gives one nonlinear equation in `hm` alone:

```
2(cL - √(g·hm)) - (hm - hR)·√( g(hm + hR) / (2·hR·hm) ) = 0
```

The code solves this with Newton's method (`_solve_hm` in
`dam_break_1d.py`). Once `hm` is known, the **bore (shock) speed** is:

```
S = um · hm / (hm - hR)
```

This whole solution is implemented in `stoker_solution()` and plotted
against the numerical result in `dam_break_comparison.png` — they should
match closely, with the numerical solution showing a bit of smearing at
the bore front (expected for a first-order finite-volume scheme).

---

## 4. What the dam structure in the visuals represents

The equations above say nothing about a physical dam — the "instant dam
break" is simply the initial condition `h(x,0) = hL` for `x < x0`,
`h(x,0) = hR` for `x > x0`, both at rest. The gray structure drawn in the
plots is a visual aid, not part of the physics:

- **Intact wall** — before `t = 0`, for context
- **Collapse animation** — a smooth, non-physical transition purely for
  legibility (the shallow water equations assume the break is
  instantaneous)
- **Broken piers + gap** — mark where the dam was, so the flow direction
  and breach location stay visually anchored as the flood wave moves
  through the domain

---

## 5. Key parameters (in `dam_break_1d.py`)

| Parameter | Meaning | Default |
|---|---|---|
| `L` | channel length [m] | 50.0 |
| `x0` | dam location [m] | 25.0 |
| `hL`, `hR` | upstream / downstream initial depth [m] | 2.0, 0.5 |
| `nx` | number of finite-volume cells | 300–500 |
| `cfl` | Courant number | 0.4 |
| `t_end` | simulation end time [s] | 4.0 |

Set `hR = 0` for a dry-bed dam break — note the current `stoker_solution()`
implements the **wet-bed** case only; the dry-bed (Ritter) solution has a
different form and would need a separate function.
