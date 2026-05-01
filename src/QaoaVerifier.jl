"""
    QaoaVerifier

Standalone verifier for QAOA satisfaction fractions on D-regular Max-k-XORSAT.

Given pre-computed QAOA angles (γ, β) and problem parameters (k, D), computes
the exact satisfaction fraction c̃(γ, β) using the Basso / Villalonga branch
tensor recurrence on the light-cone tree.

This module is a self-contained extraction of the forward evaluation pipeline
from QaoaXorsat.jl, designed for reproducibility verification of published
angle tables. It contains no optimiser, no gradients, and no GPU code.

## Public API

- `verify_satisfaction_fraction(k, D, γ, β; clause_sign=1)` — general evaluation
- `verify_xorsat(k, D, γ, β)` — Max-k-XORSAT (clause_sign = +1)
- `verify_maxcut(D, γ, β)` — MaxCut (k=2, clause_sign = -1)

## Mathematical Background

The QAOA depth-p circuit on a D-regular Max-k-XORSAT instance has a light-cone
tree with branching factor b = (D-1)(k-1). The satisfaction fraction is computed
via the branch tensor recurrence:

1. **Leaf initialisation**: B⁽⁰⁾(a) = 1 for all branch configurations a.

2. **Branch tensor step** (t = 1, …, p):
   - Compute child weights: w(a) = f(a) · B⁽ᵗ⁻¹⁾(a)
   - Constraint fold: ŵ = WHT(w), then iWHT(κ̂ · ŵ^(k-1))
   - Branching power: B⁽ᵗ⁾(a) = [fold(a)]^(D-1)

3. **Root extraction**:
   - Root message: m(a) = (-1)^{a⁰} · f(a) · B⁽ᵖ⁾(a)
   - Root kernel: κᵣ(a) = i·sin(c_s · Γ·spins(a) / 2)
   - Parity correlator: S = Σ κᵣ(a) · [m ★ … ★ m](a)  (k-fold XOR convolution)
   - Satisfaction fraction: c̃ = (1 + c_s · S) / 2

where f(a) is the mixer weight, κ is the constraint kernel, WHT is the
Walsh-Hadamard transform on (ℤ₂^n, ⊕), and c_s is the clause sign (±1).

The recurrence uses per-step max-magnitude normalisation with log-scale
accumulation to prevent Float64 overflow at high (k, D, p).

## References

- Basso, Farhi, Marwaha, Villalonga, Zhou (2022). "The Quantum Approximate
  Optimization Algorithm at High Depth for MaxCut on Large-Girth Regular Graphs
  and the Sherrington-Kirkpatrick Model." arXiv:2110.14206
- Shutty et al. (2026). arXiv:2604.24633

---
Author: John S Azariah — Centre for Quantum Software and Information, UTS
ORCID: https://orcid.org/0009-0007-9870-1970
"""
module QaoaVerifier

export verify_satisfaction_fraction, verify_xorsat, verify_maxcut

# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Types
# ══════════════════════════════════════════════════════════════════════════════

"""
    TreeParams(k, D, p)

Parameters defining a regular hypergraph QAOA light-cone tree.

- `k`: constraint arity (hyperedge size). k=2 is MaxCut.
- `D`: variable degree (regularity).
- `p`: QAOA circuit depth.
"""
struct TreeParams
    k::Int
    D::Int
    p::Int

    function TreeParams(k::Int, D::Int, p::Int)
        k ≥ 2 || throw(ArgumentError("k must be ≥ 2, got $k"))
        D ≥ 2 || throw(ArgumentError("D must be ≥ 2, got $D"))
        p ≥ 1 || throw(ArgumentError("p must be ≥ 1, got $p"))
        new(k, D, p)
    end
end

"""Branching factor per two-level step: (D-1)(k-1)."""
branching_factor(t::TreeParams) = (t.D - 1) * (t.k - 1)

"""
    QAOAAngles(γ, β)

QAOA angle parameters for depth p.

- `γ`: problem (phase-separator) angles, length p
- `β`: mixer angles, length p
"""
struct QAOAAngles{T<:Real}
    γ::Vector{T}
    β::Vector{T}

    function QAOAAngles(γ::AbstractVector{T1}, β::AbstractVector{T2}) where {T1<:Real, T2<:Real}
        length(γ) == length(β) ||
            throw(ArgumentError("γ and β must have same length, got $(length(γ)) and $(length(β))"))
        !isempty(γ) || throw(ArgumentError("need at least p=1"))
        T = promote_type(T1, T2)
        new{T}(T.(γ), T.(β))
    end
end

"""QAOA depth p."""
depth(angles::QAOAAngles) = length(angles.γ)

# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Walsh-Hadamard Transform
# ══════════════════════════════════════════════════════════════════════════════
#
# The WHT on (ℤ₂^n, ⊕) is defined as:
#   ĝ(s) = Σₓ g(x) (-1)^{⟨s,x⟩}
#
# Uses a cache-oblivious recursive decomposition with an iterative SIMD kernel
# for sub-problems that fit in L1 cache.

"""
    wht!(values)

Apply the in-place Walsh-Hadamard transform to a vector of length 2^n.
"""
function wht!(values::AbstractVector)
    N = length(values)
    N ≥ 1 || throw(ArgumentError("values must be non-empty"))
    ispow2(N) || throw(ArgumentError("length must be a power of two, got $N"))
    _wht_recursive!(values, 1, N)
    values
end

# Sub-problems at or below this size use the iterative kernel.
# 2048 ComplexF64 = 32KB, fits in L1 cache.
const _WHT_RECURSIVE_CUTOFF = 2048

function _wht_recursive!(values::AbstractVector, offset::Int, n::Int)
    if n ≤ _WHT_RECURSIVE_CUTOFF
        _wht_iterative!(values, offset, n)
        return
    end
    half = n >> 1
    @inbounds @simd for i in 0:half-1
        left = offset + i
        right = left + half
        x = values[left]
        y = values[right]
        values[left] = x + y
        values[right] = x - y
    end
    _wht_recursive!(values, offset, half)
    _wht_recursive!(values, offset + half, half)
end

function _wht_iterative!(values::AbstractVector, offset::Int, n::Int)
    block = 1
    @inbounds while block < n
        stride = 2 * block
        for base in offset:stride:(offset + n - 1)
            @simd for j in 0:(block-1)
                left = base + j
                right = left + block
                x = values[left]
                y = values[right]
                values[left] = x + y
                values[right] = x - y
            end
        end
        block = stride
    end
end

"""Out-of-place Walsh-Hadamard transform."""
wht(values::AbstractVector) = wht!(copy(values))

"""
    iwht!(values)

Apply the inverse Walsh-Hadamard transform in place: ĝ⁻¹(s) = (1/N) Σₓ g(x) (-1)^{⟨s,x⟩}.
"""
function iwht!(values::AbstractVector)
    wht!(values)
    values ./= length(values)
end

"""Out-of-place inverse Walsh-Hadamard transform."""
iwht(values::AbstractVector) = iwht!(copy(values))

"""
    xor_convolution_power(values, exponent)

Compute the repeated XOR-convolution power values ★ values ★ ⋯ ★ values
with `exponent` factors on (ℤ₂^n, ⊕).

Uses the convolution theorem: WHT(f ★ g) = WHT(f) · WHT(g).
"""
function xor_convolution_power(values::AbstractVector, exponent::Int)
    exponent ≥ 1 || throw(ArgumentError("exponent must be ≥ 1, got $exponent"))
    iwht(wht(values) .^ exponent)
end

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Branch Tensor Recurrence
# ══════════════════════════════════════════════════════════════════════════════
#
# The branch tensor B⁽ᵗ⁾ lives in the expanded (2p+1)-bit Basso branch basis
#   a = (a^[1], …, a^[p], a^[0], a^[-p], …, a^[-1])
# with |a| = 2^(2p+1) configurations.
#
# The recurrence iterates p steps from B⁽⁰⁾ = 1:
#   child_hat = WHT(f · B⁽ᵗ⁾)
#   folded    = iWHT(κ̂ · child_hat^(k-1))
#   B⁽ᵗ⁺¹⁾   = folded^(D-1)
#
# where f(a) is the mixer weight and κ(a) = cos(Γ·spins(a)/2) is the
# constraint kernel.
#
# Per-step normalisation prevents Float64 overflow at high (k, D, p):
# before each power operation, we normalise to unit max-magnitude and
# track the scale factor in log space.

"""Number of Basso branch bits: 2p + 1."""
basso_bit_count(p::Int) = 2p + 1

"""Index of the central Basso bit a^[0] in the (2p+1)-bit branch string."""
basso_root_bit_index(p::Int) = p + 1

"""Number of branch configurations: 2^(2p+1)."""
basso_configuration_count(p::Int) = one(Int) << basso_bit_count(p)

"""Z eigenvalue: 0 ↦ +1, 1 ↦ -1."""
z_eigenvalue(bit::Int) = 1 - 2bit

"""Bit positions participating in the phase computation (all except the root bit)."""
function basso_phase_bit_positions(p::Int)
    [collect(1:p); collect((p+2):(2p+1))]
end

"""
    build_gamma_full_vector(angles)

Build the full (2p+1)-length γ vector with mirrored sign convention.
Position p+1 (the root bit) has zero weight; positions 1…p carry +γ;
positions p+2…2p+1 carry -γ (mirrored).
"""
function build_gamma_full_vector(angles::QAOAAngles{T}) where T
    p = depth(angles)
    gamma_full = zeros(T, basso_bit_count(p))

    for round in 1:p
        mirror = 2p - round + 1
        bit_fwd = round          # bit position for +γ
        bit_bwd = 2p + 2 - round  # bit position for -γ
        gamma_full[bit_fwd] = angles.γ[round]
        gamma_full[bit_bwd] = -angles.γ[round]
    end

    gamma_full
end

"""
    _phase_dot(gamma_full, configuration, bit_count)

Compute Σᵢ gamma_full[i] · z_eigenvalue(bit i of configuration).
"""
@inline function _phase_dot(gamma_full::AbstractVector, configuration::Int, bit_count::Int)
    phase = zero(eltype(gamma_full))
    @inbounds for index in 1:bit_count
        phase += gamma_full[index] * z_eigenvalue((configuration >> (index - 1)) & 1)
    end
    phase
end

"""
    _fast_pow(x, n)

Specialised complex power for small positive integers (1–7), avoiding the
general x^n path which uses log/exp internally.
"""
@inline function _fast_pow(x::Complex{T}, n::Int) where T
    n == 1 && return x
    n == 2 && return x * x
    x2 = x * x
    n == 3 && return x2 * x
    n == 4 && return x2 * x2
    x4 = x2 * x2
    n == 5 && return x4 * x
    n == 6 && return x4 * x2
    n == 7 && return x4 * x2 * x
    return x ^ n
end

"""
    basso_trig_table(angles)

Precompute the trigonometric factors for the mixer weight f(a).
Returns a 2×2p matrix: row 1 = cos(β), row 2 = i·sin(β) for each transition,
with the mirrored angle convention.
"""
function basso_trig_table(angles::QAOAAngles{T}) where T
    p = depth(angles)
    CT = Complex{T}
    trigs = zeros(CT, 2, 2p)

    for round in 1:p
        mirror = 2p - round + 1
        β = angles.β[round]

        trigs[1, round]  = cos(β)
        trigs[2, round]  = complex(zero(T), sin(β))
        trigs[1, mirror] = cos(-β)
        trigs[2, mirror] = complex(zero(T), sin(-β))
    end

    trigs
end

"""
    _compute_f_table(trig_table, bit_count, N, T)

Compute the mixer weight f(a) for all N = 2^(2p+1) configurations.

f(a) = ½ · ∏ⱼ₌₁²ᵖ [ cos(βⱼ) if aⱼ = aⱼ₊₁,  i·sin(βⱼ) if aⱼ ≠ aⱼ₊₁ ]

where the product runs over all 2p transitions between adjacent bits in
the (2p+1)-bit branch string, using the mirrored β convention.
"""
function _compute_f_table(trig_table::Matrix{Complex{T}}, bit_count::Int, N::Int, ::Type{T}) where T
    table = Vector{Complex{T}}(undef, N)
    transitions = bit_count - 1  # = 2p
    for config in 0:N-1
        weight = complex(one(T) / 2)
        @inbounds for j in 1:transitions
            d = xor((config >> (j - 1)) & 1, (config >> j) & 1)
            weight *= trig_table[d + 1, j]
        end
        @inbounds table[config + 1] = weight
    end
    table
end

"""
    basso_root_parity(configuration, p)

Extract the parity sign (-1)^{a⁰} from the root bit of a branch configuration.
"""
function basso_root_parity(configuration::Int, p::Int)
    root_bit = basso_root_bit_index(p)
    z_eigenvalue((configuration >> (root_bit - 1)) & 1)
end

"""
    _evaluate_normalized(params, angles; clause_sign=1)

Run the full normalised Basso evaluation pipeline.

Returns (value, log_total_scale, S_normalized) where:
- value is the satisfaction fraction c̃
- log_total_scale is the accumulated log-scale factor
- S_normalized is the normalised parity correlator
"""
function _evaluate_normalized(
    params::TreeParams,
    angles::QAOAAngles{T};
    clause_sign::Int = 1,
) where T
    p = params.p
    k = params.k
    D = params.D
    arity = k - 1     # child arity in constraint fold
    degree = D - 1     # branching degree (power after fold)
    bit_count = basso_bit_count(p)
    N = basso_configuration_count(p)

    depth(angles) == p || throw(ArgumentError(
        "angle depth $(depth(angles)) must match tree depth $p"))
    clause_sign in (-1, 1) || throw(ArgumentError(
        "clause_sign must be ±1, got $clause_sign"))

    # ── Precompute angle-dependent tables ──────────────────────────────────

    gamma_full = build_gamma_full_vector(angles)
    trig_table = basso_trig_table(angles)
    f_table = _compute_f_table(trig_table, bit_count, N, T)

    # Phase arguments and constraint kernel: κ(a) = cos(Γ·spins(a)/2)
    half = one(T) / 2
    phase_args = Vector{T}(undef, N)
    kernel = Vector{Complex{T}}(undef, N)
    for config in 0:N-1
        ph = _phase_dot(gamma_full, config, bit_count)
        @inbounds phase_args[config + 1] = ph
        @inbounds kernel[config + 1] = complex(cos(half * ph))
    end
    kernel_hat = wht!(copy(kernel))

    # ── Branch tensor iteration with per-step normalisation ────────────────
    #
    # Before each power operation, normalise to unit max-magnitude if the
    # magnitude exceeds a safety threshold, and track the scale in log space.
    # This prevents Float64 overflow at high (k, D, p) while preserving
    # precision when normalisation is unnecessary.

    _NORM_THRESHOLD = T(1e30)

    scratch = Vector{Complex{T}}(undef, N)
    B = ones(Complex{T}, N)      # B⁽⁰⁾ = 1
    log_s = zero(T)              # accumulated log-scale on B

    for t in 1:p
        # child_weights = f_table .* B, then WHT in-place
        @inbounds @simd for i in 1:N
            scratch[i] = f_table[i] * B[i]
        end
        wht!(scratch)

        # Normalise child_hat before ^(k-1) if needed
        ch_scale = maximum(abs, scratch)
        if ch_scale > _NORM_THRESHOLD
            inv_ch = one(T) / ch_scale
            @inbounds @simd for i in 1:N
                scratch[i] *= inv_ch
            end
        else
            ch_scale = one(T)
        end

        # folded = iWHT(kernel_hat .* child_hat^arity)
        @inbounds @simd for i in 1:N
            scratch[i] = kernel_hat[i] * _fast_pow(scratch[i], arity)
        end
        iwht!(scratch)

        # Normalise folded before ^(D-1) if needed
        fld_scale = maximum(abs, scratch)
        if fld_scale > _NORM_THRESHOLD
            inv_fld = one(T) / fld_scale
            @inbounds @simd for i in 1:N
                scratch[i] *= inv_fld
            end
        else
            fld_scale = one(T)
        end

        # B⁽ᵗ⁺¹⁾ = folded^degree
        @inbounds @simd for i in 1:N
            B[i] = _fast_pow(scratch[i], degree)
        end

        # Update log-scale recurrence:
        #   log_s[t+1] = (k-1)(D-1) · log_s[t] + (k-1)·log(α_t) + (D-1)·log(β_t)
        log_s = arity * degree * log_s +
                arity * log(ch_scale) +
                degree * log(fld_scale)
    end

    # ── Root extraction ────────────────────────────────────────────────────
    #
    # Root message: m(a) = (-1)^{a⁰} · f(a) · B⁽ᵖ⁾(a)
    # Root kernel:  κᵣ(a) = i·sin(c_s · Γ·spins(a) / 2)
    # Parity correlator: S = Σ κᵣ(a) · [m ★ … ★ m](a)  (k-fold convolution)
    # Satisfaction fraction: c̃ = (1 + c_s · S) / 2

    root_msg = Vector{Complex{T}}(undef, N)
    for config in 0:N-1
        parity = basso_root_parity(config, p)
        @inbounds root_msg[config + 1] = parity * f_table[config + 1] * B[config + 1]
    end

    cs = T(clause_sign)
    root_kernel = Vector{Complex{T}}(undef, N)
    for config in 0:N-1
        @inbounds root_kernel[config + 1] = complex(zero(T), sin(half * cs * phase_args[config + 1]))
    end

    # Root fold with normalisation: WHT(root_msg), normalise, raise to ^k
    msg_hat = wht!(complex.(root_msg))
    mh_scale = maximum(abs, msg_hat)
    if mh_scale > _NORM_THRESHOLD
        msg_hat .*= one(T) / mh_scale
    else
        mh_scale = one(T)
    end
    msg_hat_power = msg_hat .^ k
    conv = iwht(msg_hat_power)
    S_normalized = sum(root_kernel .* conv)

    # Total log-scale
    log_total_scale = k * (log_s + log(mh_scale))

    # Compute physical value using log-space multiplication
    re_S_norm = real(S_normalized)
    if re_S_norm == 0 || !isfinite(log_total_scale)
        value = half
    else
        log_product = log_total_scale + log(abs(re_S_norm))
        if log_product > 700  # would overflow exp()
            value = T(NaN)
        else
            scaled_re_S = copysign(exp(log_product), re_S_norm)
            value = (1 + clause_sign * scaled_re_S) / 2
        end
    end

    (value = T(value), log_total_scale = log_total_scale, S_normalized = S_normalized)
end

# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Public API
# ══════════════════════════════════════════════════════════════════════════════

"""
    verify_satisfaction_fraction(k, D, γ, β; clause_sign=1) -> Float64

Compute the exact QAOA satisfaction fraction c̃(γ, β) for the root clause
of a D-regular Max-k-XORSAT instance at QAOA depth p = length(γ).

# Arguments
- `k::Int`: constraint arity (hyperedge size). k=2 for MaxCut.
- `D::Int`: variable degree (graph regularity).
- `γ::Vector{Float64}`: problem (phase-separator) angles, length p.
- `β::Vector{Float64}`: mixer angles, length p.
- `clause_sign::Int=1`: +1 for XORSAT clause (1+Z₁⋯Zₖ)/2,
                         -1 for MaxCut clause (1-Z₁Z₂)/2.

# Returns
The satisfaction fraction c̃ ∈ [0, 1].

# Examples
```julia
using QaoaVerifier

# MaxCut on 3-regular graph, depth 1
c̃ = verify_satisfaction_fraction(2, 3,
    [5.667705597838953], [1.1780972446407532];
    clause_sign = -1)
# c̃ ≈ 0.6925

# Max-3-XORSAT on 4-regular, depth 1
c̃ = verify_satisfaction_fraction(3, 4,
    [3.6916431695847245], [0.28737134801591857])
# c̃ ≈ 0.6761
```
"""
function verify_satisfaction_fraction(
    k::Int, D::Int,
    γ::AbstractVector{<:Real}, β::AbstractVector{<:Real};
    clause_sign::Int = 1,
)::Float64
    params = TreeParams(k, D, length(γ))
    angles = QAOAAngles(Float64.(γ), Float64.(β))
    result = _evaluate_normalized(params, angles; clause_sign)
    Float64(result.value)
end

"""
    verify_maxcut(D, γ, β) -> Float64

Compute the exact QAOA satisfaction fraction for MaxCut (k=2, clause_sign=-1)
on a D-regular graph.

Equivalent to `verify_satisfaction_fraction(2, D, γ, β; clause_sign=-1)`.
"""
function verify_maxcut(D::Int, γ::AbstractVector{<:Real}, β::AbstractVector{<:Real})::Float64
    verify_satisfaction_fraction(2, D, γ, β; clause_sign = -1)
end

"""
    verify_xorsat(k, D, γ, β) -> Float64

Compute the exact QAOA satisfaction fraction for Max-k-XORSAT (clause_sign=+1)
on a D-regular hypergraph.

Equivalent to `verify_satisfaction_fraction(k, D, γ, β; clause_sign=1)`.
"""
function verify_xorsat(k::Int, D::Int, γ::AbstractVector{<:Real}, β::AbstractVector{<:Real})::Float64
    verify_satisfaction_fraction(k, D, γ, β; clause_sign = 1)
end

end # module QaoaVerifier
