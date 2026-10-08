# The trainable meta-GGA EXCHANGE functional: LDA exchange x MLP enhancement F_x,θ,
# implementing DFTK's native DftFunctionals.Functional{:mgga,:x} interface (the same
# interface PbeExchange uses for :gga) so it plugs straight into Xc([...]) and
# differentiates through the whole SCF via ForwardDiff - no custom Reactant/Enzyme term
# needed.
#
#   E_x[ρ,τ] = ∫ e_x(ρ) · F_x,θ(u, w) dr,   F_x,θ = 1 + N_θ(u, w) - N_θ(0, 0)
#
# enhancement over LDA exchange e_x = C_X ρ^(4/3), with N_θ a small MLP over bounded
# features that are ZERO at the homogeneous electron gas:
#     u = s²/(1+s²) ∈ [0,1)              (the B97 variable)
#     w = (τ_unif-τ)/(τ_unif+τ) ∈ (-1,1] (τ_W-free Becke/Minnesota variable)
# Subtracting the network's own value at the HEG point makes the HEG limit exact for ANY
# θ, with the correct leading orders of approach. Exchange obeys uniform scaling, so
# there is no rs input.
#
# Unpolarized only (n_spin=1): DftFunctionals' packed σ = (σ_αα, σ_αβ, σ_ββ) format does
# not expose per-channel ∇ρ_σ, which the spin-scaling relation E_x[ρ↑,ρ↓] = ½Σ_σ E_x[2ρ_σ]
# needs; the training dataset (Si) is unpolarized, so this is not a restriction in practice.

using ComponentArrays: ComponentArray
using DftFunctionals
using DftFunctionals: Functional, arithmetic_type
using ForwardDiff
using Lux
using Lux.LuxCore: statelength, stateless_apply
using Random

# --- Continuum model -----------------------------------------------------------

"""LDA exchange energy density e_x(ρ) = C_X ρ^(4/3)."""
const C_X = -(3/4) * (3/π)^(1/3)
lda_x_energy_density(ρ) = C_X * ρ^(4/3)

"""Density floor: keeps ρ^(4/3)/ρ^(1/3) and the s division finite. Small enough
to never affect the energy."""
const ρ_floor = 1e-30

"""Squared reduced gradient s² = |∇ρ|² / (2 k_F ρ)² = σ / (2 k_F ρ)²."""
reduced_gradient_sq(ρ, σ) = σ / (4 * (3π^2)^(2/3) * ρ^(8/3))

"""Bounded inhomogeneity feature u = s²/(1+s²) ∈ [0,1): 0 at the HEG, → 1 at
large reduced gradient s."""
function feature_inhomogeneity(ρ, σ)
    s2 = reduced_gradient_sq(ρ, σ)
    s2 / (1 + s2)
end

"""Uniform-gas kinetic energy density τ_unif = (3/10)(3π²)^(2/3) ρ^(5/3)."""
tau_unif(ρ) = (3/10) * (3π^2)^(2/3) * ρ^(5/3)

"""Bounded kinetic-energy feature w = (τ_unif - τ)/(τ_unif + τ): 0 at the HEG,
+1 in density tails (τ→0), → -1 in the strongly overlapping/metallic limit.
τ_W-free, so no clamp is needed and the denominator never vanishes."""
feature_w(ρ, τ) = (tau_unif(ρ) - τ) / (tau_unif(ρ) + τ)

"""
E_x[ρ,τ] = ∫ e_x · (1 + N_θ(u, w) - N_θ(0, 0)) dr, implementing DFTK's native
Functional{:mgga,:x} interface.

A Lux wrapper layer around the MLP N_θ: initialize the trainable parameters canonically
with `ps, _ = Lux.setup(rng, mlp)`. The functional must be a pure function of (ρ, σ, τ),
so the network has to be stateless.
"""
struct MGGAExchange{L,CA<:ComponentArray} <: Functional{:mgga,:x}
    mlp::L
    parameters::CA
    identifier::Symbol
    function MGGAExchange(mlp::L, parameters::CA; identifier=:mgga_x_learned) where {L,CA<:ComponentArray}
        @assert iszero(statelength(mlp)) "XC models must be stateless"
        new{L,CA}(mlp, parameters, identifier)
    end
end

DftFunctionals.identifier(f::MGGAExchange) = f.identifier
DftFunctionals.parameters(f::MGGAExchange) = f.parameters
DftFunctionals.needs_τ(::MGGAExchange)     = true
function DftFunctionals.change_parameters(f::MGGAExchange, parameters::ComponentArray;
                                          keep_identifier=false)
    MGGAExchange(f.mlp, parameters;
                identifier=keep_identifier ? f.identifier : :mgga_x_learned_custom)
end

"""MLP over (u, w). The last layer is zero-initialized so F_x ≡ 1 (LDA exchange) at
start; the HEG anchor keeps the limit exact for any later θ."""
function mlp_architecture(; hidden_sizes=(8,))
    sizes = (2, hidden_sizes...)
    Chain([Dense(sizes[i], sizes[i+1], softplus) for i in 1:length(hidden_sizes)]...,
         Dense(last(sizes), 1; init_weight=zeros64, init_bias=zeros64))
end

function MGGAExchange(; hidden_sizes=(8,), rng=Xoshiro(0))
    mlp = mlp_architecture(; hidden_sizes)
    ps, _ = Lux.setup(rng, mlp)
    MGGAExchange(mlp, ComponentArray(Lux.f64(ps)))
end

"""Rebuild with the SAME architecture (hidden_sizes) but new parameter values - used by
the training loop, which only ever carries the flat θ (a ComponentArray, or one promoted
to Dual by ForwardDiff), not the Lux layer object."""
MGGAExchange(θ::ComponentArray; hidden_sizes=(8,)) =
    MGGAExchange(mlp_architecture(; hidden_sizes), θ)

"""F over a single feature vector x = (u, w): 1 + N(x) - N(0)."""
function anchored_enhancement(f::MGGAExchange, x::AbstractVector, ps)
    N  = only(stateless_apply(f.mlp, reshape(x, :, 1), ps))
    N0 = only(stateless_apply(f.mlp, zero(reshape(x, :, 1)), ps))
    # TODO that is a correct but inefficient way to enforce HEG
    #      since it doubles the total NN evaluations; could potentially be
    #      dropped, or replaced by a grid-free loss term with high weight
    1 + N - N0
end

"""F_x,θ(fs, fα) on the bounded (fs, fα) features, the signature shared with
diagnostics/plots; fα is converted to w internally."""
function enhancement_factor(f::MGGAExchange, fs, fα, ps)
    t = fα / (1 - fα) + (5 / 3) * fs / (1 - fs)
    w = (1 - t) / (1 + t)
    anchored_enhancement(f, [fs, w], ps)
end

"""Per-point exchange energy density, the scalar function the potential_terms fallback
below differentiates via ForwardDiff. ρ, σ, τ are scalars (one grid point, the single
unpolarized spin channel)."""
function energy(f::MGGAExchange, ρ::Number, σ::Number, τ::Number, ps)
    ρs = max(ρ, ρ_floor)
    u  = feature_inhomogeneity(ρs, max(σ, zero(σ)))
    w  = feature_w(ρs, τ)
    lda_x_energy_density(ρs) * anchored_enhancement(f, [u, w], ps)
end

# No :mgga auto-derive fallback exists upstream (only :gga does), so this differentiates
# the scalar energy above by hand, per grid point.

function DftFunctionals.potential_terms(func::MGGAExchange, ρ::AbstractMatrix{T},
                                        σ::AbstractMatrix{U}, τ::AbstractMatrix{V}, args...) where {T,U,V}
    size(ρ, 1) == 1 || error("MGGAExchange only supports unpolarized densities (n_spin=1)")
    n_p = size(ρ, 2)
    TT  = arithmetic_type(func, T, U, V)

    # e is (1, n_p), matching Libxc's own convention: potential_terms from multiple
    # functionals get elementwise-summed (mergesum).
    e  = similar(ρ, TT, 1, n_p)
    Vρ = similar(ρ, TT, 1, n_p)
    Vσ = similar(ρ, TT, 1, n_p)
    Vτ = similar(ρ, TT, 1, n_p)
    @views for i = 1:n_p
        potential_terms!(e[:, i], Vρ[:, i], Vσ[:, i], Vτ[:, i], func, ρ[1, i], σ[1, i], τ[1, i])
    end
    (; e, Vρ, Vσ, Vτ)
end
function potential_terms!(e, Vρ, Vσ, Vτ, func::MGGAExchange, ρ::T, σ::U, τ::V) where {T,U,V}
    ps = func.parameters
    g  = ForwardDiff.gradient(x -> energy(func, x[1], x[2], x[3], ps), [ρ, σ, τ])
    e[1]  = energy(func, ρ, σ, τ, ps)
    Vρ[1] = g[1]
    Vσ[1] = g[2]
    Vτ[1] = g[3]
    nothing
end

# Energy-only path (e.g. compute_consistent_energies) - no differentiation needed here.

function DftFunctionals.energy_density(func::MGGAExchange, ρ::AbstractMatrix,
                                       σ::AbstractMatrix, τ::AbstractMatrix, args...)
    size(ρ, 1) == 1 || error("MGGAExchange only supports unpolarized densities (n_spin=1)")
    ps = func.parameters
    reshape([energy(func, ρ[1, i], σ[1, i], τ[1, i], ps) for i in axes(ρ, 2)], 1, :)  # (1, n_p)
end
