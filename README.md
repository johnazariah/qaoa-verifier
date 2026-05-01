# QAOA Satisfaction Fraction Verifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Companion verification code for Shutty et al. (arXiv:2604.24633).

Given pre-computed QAOA angles (γ, β) and problem parameters (k, D), computes
the **exact** satisfaction fraction c̃(γ, β) for QAOA on D-regular Max-k-XORSAT.
This is a standalone, self-contained extraction of the forward evaluation
pipeline from [QaoaXorsat.jl](https://github.com/johnazariah/qaoa-xorsat) — the
full research codebase that also includes angle optimisation, gradient
computation, GPU acceleration, and spectral analysis.

**Author**: John S Azariah — Centre for Quantum Software and Information, UTS
**ORCID**: [0009-0007-9870-1970](https://orcid.org/0009-0007-9870-1970)

---

## Quick Start

### Prerequisites

- **Julia 1.9 or later** — download from [julialang.org](https://julialang.org/downloads/)

No other dependencies are required. The verifier uses only the Julia standard
library.

### Setup

```bash
# Clone the repository
git clone https://github.com/johnazariah/qaoa-verifier.git
cd qaoa-verifier

# Install dependencies (there are none beyond stdlib, but this validates the environment)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Verify published results

```bash
# Run the verifier against the reference angle table
julia --project=. verify.jl data/reference-results.csv
```

Expected output:

```
╔══════════════════════════════════════════════════════════════════╗
║  QAOA Satisfaction Fraction Verifier                            ║
║  Companion to Shutty et al. (arXiv:2604.24633)                 ║
╚══════════════════════════════════════════════════════════════════╝

Reading angles from: data/reference-results.csv

Found 21 configurations to verify.
──────────────────────────────────────────────────────────────────────────────────────────
  (k=2, D=3, p= 1) [MaxCut          ]  c̃=0.692450089730  claimed=0.692450089730  OK (Δ=...)  [0.0s]
  (k=2, D=3, p= 2) [MaxCut          ]  c̃=0.755906458453  claimed=0.755906458453  OK (Δ=...)  [0.0s]
  ...
  (k=3, D=4, p= 8) [Max-3-XORSAT    ]  c̃=0.854102061491  claimed=0.854102061491  OK (Δ=...)  [1.2s]
  ...
──────────────────────────────────────────────────────────────────────────────────────────
✓  All evaluations completed successfully.
```

### Verify a single configuration from the command line

```bash
# MaxCut on 3-regular graph, depth 1
julia --project=. verify.jl \
    --k 2 --D 3 \
    --gamma "5.667705597838953" \
    --beta "1.1780972446407532" \
    --clause-sign -1

# Max-3-XORSAT on 4-regular, depth 3
julia --project=. verify.jl \
    --k 3 --D 4 \
    --gamma "2.780197925127931;3.7658412617389994;3.870022135985064" \
    --beta "1.9545683188631753;0.2954713531707407;0.17591776293047443" \
    --ctilde 0.777144062579
```

### Run the test suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This validates the verifier against known reference results for MaxCut (k=2, D=3)
and Max-3-XORSAT (k=3, D=4,5) at depths p=1 through p=8.

---

## Using as a Julia package

```julia
using QaoaVerifier

# MaxCut on 3-regular graph, depth 1
c̃ = verify_maxcut(3,
    [5.667705597838953],
    [1.1780972446407532])
# c̃ ≈ 0.6925

# Max-3-XORSAT on 4-regular, depth 1
c̃ = verify_xorsat(3, 4,
    [3.6916431695847245],
    [0.28737134801591857])
# c̃ ≈ 0.6761

# General interface with explicit clause sign
c̃ = verify_satisfaction_fraction(k, D, γ, β; clause_sign=±1)
```

### API

| Function | Arguments | Description |
|----------|-----------|-------------|
| `verify_satisfaction_fraction(k, D, γ, β; clause_sign)` | k: arity, D: degree, γ/β: angle vectors | General evaluation |
| `verify_maxcut(D, γ, β)` | D: degree, γ/β: angle vectors | MaxCut (k=2, clause_sign=-1) |
| `verify_xorsat(k, D, γ, β)` | k: arity, D: degree, γ/β: angle vectors | XORSAT (clause_sign=+1) |

All functions return a `Float64` satisfaction fraction c̃ ∈ [0, 1].

---

## What This Computes

### The QAOA Circuit on the Light-Cone Tree

For a D-regular Max-k-XORSAT instance, the depth-p QAOA circuit's expected
satisfaction of any single clause depends only on the local structure within
the light cone — a tree with branching factor b = (D−1)(k−1) extending p
levels from the root clause.

The evaluation avoids constructing or simulating the exponentially large
quantum state. Instead, it uses a **transfer-matrix / branch-tensor recurrence**
that contracts the tree from leaves to root in O(4^p) space and
O(p · 4^p · log(4^p)) time (dominated by the Walsh-Hadamard transforms).

### The Branch Tensor Recurrence

The algorithm proceeds in three stages:

**Stage 1: Leaf initialisation.** The branch tensor B⁽⁰⁾ is initialised to
the all-ones vector over 2^(2p+1) configurations. Each configuration is a
(2p+1)-bit string a = (a^[1], …, a^[p], a^[0], a^[-p], …, a^[-1]) encoding
the ket and bra indices at each depth level plus a root spin.

**Stage 2: Iterative contraction** (t = 1, …, p). At each step:

1. Form the **child weights** w(a) = f(a) · B^(t−1)(a), where f(a) is the
   mixer weight — a product of cos(β) and i·sin(β) factors determined by
   the bit transitions in the branch string.

2. Apply the **constraint fold**: use the Walsh-Hadamard transform to
   efficiently compute the k-body XOR convolution
   ŵ = WHT(w), then iWHT(κ̂ · ŵ^(k−1)), where κ(a) = cos(Γ·spins(a)/2)
   is the constraint kernel encoding the phase-separator gate.

3. Raise to the **(D−1)th power** to account for the branching degree:
   B^(t)(a) = [fold(a)]^(D−1).

To prevent Float64 overflow at high (k, D, p), each power operation is
preceded by max-magnitude normalisation with the scale factor tracked in
log space. The accumulated scale is applied once at the root.

**Stage 3: Root extraction.** The parity correlator ⟨Z₁···Zₖ⟩ is computed
by forming the root message m(a) = (−1)^{a⁰} · f(a) · B^(p)(a), applying a
k-fold XOR convolution via WHT, and contracting against the root kernel
κᵣ(a) = i·sin(c_s · Γ·spins(a)/2). The satisfaction fraction is then

> c̃ = (1 + c_s · ⟨Z₁···Zₖ⟩) / 2

where c_s = +1 for XORSAT and c_s = −1 for MaxCut.

### Computational Cost

| Parameter | Cost |
|-----------|------|
| Space | O(4^p) complex numbers ≈ 16 · 4^p bytes |
| Time per step | O(4^p · p) for the WHT |
| Total time | O(p² · 4^p) |
| p=8 | ~1 second |
| p=12 | ~1 minute |
| p=16 | ~30 minutes (estimated) |

---

## Input Format

### CSV file

```csv
# Comment lines start with #
k,D,p,ctilde,source,gamma,beta
2,3,1,0.692450089730,maxcut-sweep,5.667705597838953,1.1780972446407532
3,4,2,0.739122784672,composite-best,2.719;3.863,1.934;0.211
```

- `k`: constraint arity (2 for MaxCut, 3+ for XORSAT)
- `D`: variable degree (graph regularity)
- `p`: QAOA depth
- `ctilde`: claimed satisfaction fraction (for verification)
- `source`: provenance label (informational only)
- `gamma`: semicolon-separated γ angles (γ₁;γ₂;…;γₚ)
- `beta`: semicolon-separated β angles (β₁;β₂;…;βₚ)

The clause sign is inferred from k: k=2 uses clause_sign=−1 (MaxCut),
k≥3 uses clause_sign=+1 (XORSAT).

### Command-line arguments

```
--k K              Constraint arity
--D D              Variable degree
--gamma γ₁;γ₂;…   Problem angles (semicolon-separated)
--beta β₁;β₂;…    Mixer angles (semicolon-separated)
--clause-sign ±1   Optional (default: -1 for k=2, +1 for k≥3)
--ctilde VALUE     Optional claimed value for comparison
```

---

## Repository Structure

```
qaoa-verifier/
├── Project.toml            # Julia package metadata (no external dependencies)
├── README.md               # This file
├── LICENSE                  # MIT License
├── CITATION.cff            # Citation metadata
├── verify.jl               # CLI entry point — reads CSV or args, prints results
├── src/
│   └── QaoaVerifier.jl     # The module (~400 lines, fully self-contained)
├── test/
│   └── runtests.jl         # Validation against known reference results
├── data/
│   └── reference-results.csv  # Published angles and satisfaction fractions
└── results/
    └── (generated)         # Output from verification runs
```

### Source code overview (~400 lines)

The module `src/QaoaVerifier.jl` is organised into four sections:

1. **Types** (~20 lines): `TreeParams(k, D, p)` and `QAOAAngles(γ, β)`.
2. **Walsh-Hadamard Transform** (~60 lines): Cache-oblivious recursive WHT
   with iterative SIMD kernel for L1-resident sub-problems.
3. **Branch Tensor Recurrence** (~200 lines): Mixer weight table, constraint
   kernel, WHT-accelerated fold, normalised power iteration.
4. **Public API** (~30 lines): `verify_satisfaction_fraction`, `verify_maxcut`,
   `verify_xorsat`.

---

## What This Is NOT

- **Not an optimiser.** It doesn't search for angles — it evaluates given angles.
- **Not a research tool.** No gradient computation, spectral analysis, or GPU support.
- **Not a dependency of the main repo.** It's a frozen snapshot of the evaluator
  for reproducibility.
- **Not trying to be fast.** Single-threaded, no SIMD tricks beyond what the
  compiler provides. Correctness and clarity over performance. (It's still fast:
  p=12 evaluates in about a minute.)

---

## Citation

If you use this code, please cite:

```bibtex
@software{azariah2026qaoa_verifier,
  author    = {Azariah, John S},
  title     = {{QAOA} Satisfaction Fraction Verifier},
  year      = {2026},
  url       = {https://github.com/johnazariah/qaoa-verifier},
  note      = {Companion code for Shutty et al. (arXiv:2604.24633)}
}
```

## License

MIT — see [LICENSE](LICENSE).
