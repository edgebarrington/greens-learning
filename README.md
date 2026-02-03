# Learning Green’s Functions as Operators

This project studies **operator learning for elliptic PDEs** through the lens of Green’s functions.  
Rather than learning solutions pointwise, we learn a **kernel representation of the inverse operator** and train it *only* through its action on forcing functions.

The goal is not raw accuracy at all costs, but **modeling clarity**:
- What does the model learn?
- What physical structure must be enforced?
- Where and why does it fail?

The project is intentionally physics-forward, math-heavy, and avoids black-box ML shortcuts.

---

## Problem Setting

We consider the 1D Poisson problem

\[
- u''(x) = f(x), \quad x \in (0,1), \quad u(0)=u(1)=0.
\]

The exact solution can be written using a Green’s function:
\[
u(x) = \int_0^1 G(x,x') f(x') \, dx'.
\]

Instead of learning \(u\) directly, we aim to **learn the Green’s function \(G\)** as a kernel, so that the solution emerges through operator application.

---

## Project Structure
- greens-learning/
├── solvers/ # Analytic, finite-difference, and spectral solvers
├──data/ # Forcing generation, datasets, loaders
├── models/ # Kernel parameterizations and constrained variants
├── training/ # Operator loss and training logic
├── experiments/ # Training runs, evaluation, ablations
└── results/ # Recorded metrics (no model checkpoints)

---

## Phase 1 — Physics Baselines

We first implement and cross-validate three classical solvers:

- **Analytic Green’s function**
- **Finite difference solver**
- **Spectral (sine-series) solver**

All three agree to \(O(10^{-4})\)–\(O(10^{-14})\), establishing a trusted numerical foundation.

This phase ensures that any later ML behavior can be diagnosed against known physics.

---

## Phase 2 — Data Generation

We generate datasets of forcing–solution pairs \((f, u)\), where:
- Forcings are sampled as truncated sine series
  \[
  f(x) = \sum_{k=1}^K a_k \sin(k\pi x), \quad a_k \sim \mathcal N(0, k^{-p})
  \]
- Solutions are computed using the **spectral solver** as ground truth.

This produces smooth, interpretable data with controlled frequency content and fixed train/val/test splits.

---

## Phase 3 — Learning the Operator

### Kernel Parameterization

We parameterize a kernel
\[
g_\theta(x, x')
\]
using a small MLP. This kernel is never supervised directly.

### Physics Constraints (Architectural)

We enforce physical structure **by construction**, not via penalties:

- **Symmetry**
  \[
  G(x,x') = G(x',x)
  \]
- **Dirichlet boundary conditions**
  \[
  G(0,x') = G(1,x') = 0
  \]

The learned kernel takes the form
\[
\hat G_\theta(x,x') =
\frac{1}{2}[g_\theta(x,x') + g_\theta(x',x)] \, x(1-x)x'(1-x').
\]

---

### Operator Loss

The model is trained only through operator action:

\[
\hat u(x_i) = \sum_j \hat G_\theta(x_i,x_j) f(x_j)\, h
\]

and minimizes
\[
\mathcal L = \|\hat u - u\|_2^2.
\]

Notably:
- The true Green’s function is never shown to the model.
- No PDE residuals or collocation losses are used.

---

## Phase 4 — Evaluation & Diagnostics

### Learned vs Analytic Green’s Function

The learned kernel reproduces the **global structure** of the analytic Green’s function:
- symmetry
- zero boundaries
- correct triangular form

The largest discrepancies occur near the diagonal \(x=x'\), where the true Green’s function has limited smoothness.

---

### Validation Operator Performance

On unseen forcings:
- Median relative solution error is low
- A heavier tail appears for high-frequency forcings
- Errors are structured, not random

This confirms that the model learns a meaningful inverse operator, not just training memorization.

---

## Ablation Studies

We perform controlled ablations to understand inductive biases:

### 1. No Symmetry (BCs kept)
- Improves empirical solution accuracy
- Violates a fundamental physical property
- Demonstrates a bias–variance tradeoff between physics and expressivity

### 2. No Boundary Conditions (Symmetry kept)
- Produces clear boundary leakage
- Significantly worsens validation error
- Confirms that BCs must be enforced architecturally

These ablations show that **not all physics constraints behave the same**:
- Some trade accuracy for interpretability
- Others are non-negotiable for correctness

---

## Key Takeaways

- Operator learning can recover Green’s-function structure without direct supervision.
- Architectural constraints are more reliable than loss penalties.
- Validation failures are interpretable and tied to frequency content and diagonal singularity.
- Physical correctness and empirical accuracy can be in tension — and that tension is informative.

---

## Future Work

- Diagonal-aware kernel parameterizations
- Frequency-conditioned kernels
- Extension to variable-coefficient or higher-dimensional problems

---

## Motivation

This project was built to demonstrate:
- modeling intuition over benchmark chasing
- clean numerical reasoning
- principled use of ML where it adds value

It is intended for readers comfortable with PDEs, numerical analysis, and modern ML.
