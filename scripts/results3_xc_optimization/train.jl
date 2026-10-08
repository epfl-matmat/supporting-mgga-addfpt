# Run: julia +1.12.5 --project train.jl <config.jl> <rundir> [num_train_steps]
#
# Self-consistent XC training on silicon with mGGA AD-DFPT
#
# A run is a directory holding one JLD2 snapshot (θ, μ, iteration) per accepted BFGS
# iterate. Re-running the same rundir resumes from the latest snapshot.
using ComponentArrays
using DFTK
using ForwardDiff
using JLD2
using LinearAlgebra: dot, norm
using LineSearches
using Optim
using Printf
using Random

include("input_pipeline.jl")
include("xc_model.jl")

DFTK.setup_threading()

snapshot_path(workdir, step) = joinpath(workdir, @sprintf("bfgs_%04d.jld2", step))
snapshot_iter(path) = parse(Int, match(r"bfgs_(\d+)\.jld2$", path).captures[1])
snapshot_files(workdir) =
    sort(filter(f -> occursin(r"bfgs_\d+\.jld2$", f), readdir(workdir; join=true)))

const DEFAULT_CONFIG = (;
    ref_functional  = "hse06",
    base_functional = [:gga_c_pbe],   # fixed correlation added alongside the learned exchange
    trainset        = [("diamond", 0), ("diamond", 3), ("diamond", 6)],
    loss_weights    = (; E=10.0, ρ=1.0, τ=1.0),
    tol_scf         = 1e-10,
    hidden_sizes    = (8,),
)
get_config() = DEFAULT_CONFIG

# --- Reference data -------------------------------------------------------------

struct RefCell
    name::String
    lattice::Matrix{Float64}
    atoms::Vector{ElementPsp}
    positions::Vector{Vector{Float64}}
    Ecut::Float64
    kgrid::NTuple{3,Int}
    temperature::Float64
    smearing::Smearing.SmearingFunction
    fft_size::NTuple{3,Int}
    n_atoms::Int
    E_ref::Float64
    ρ_ref::Array{Float64,4}
    τ_ref::Array{Float64,4}
end

function load_ref_cells(config, names)
    bundle = bundle_path(config.ref_functional)
    psps   = load_pseudopotentials(bundle)
    map(names) do name
        cell = load_cell(bundle, name)
        lattice, atoms, positions = dftk_structure(cell, psps)
        (; temperature, smearing) = parse_smearing(cell)
        RefCell(name, lattice, atoms, positions, cell.ecutwfc, cell.mp_grid,
               temperature, smearing, cell.fft_size, length(cell.atomic_structure.atoms),
               cell.qe_energy_terms_Ha.etot, cell.ρ, cell.τ)
    end
end

function build_basis(config, rc::RefCell, θ)
    T = promote_type(eltype(θ), eltype(rc.lattice))
    functional = MGGAExchange(θ; config.hidden_sizes)
    model = model_atomic(Matrix{T}(rc.lattice), rc.atoms, rc.positions;
                         rc.temperature, rc.smearing,
                         extra_terms=[Hartree(), Xc([functional, config.base_functional...])])
    PlaneWaveBasis(model; rc.Ecut, rc.kgrid, fft_size=rc.fft_size)
end

# --- Loss -------------------------------------------------------------------------

function loss_channels(E, ρ, τ, μ, rc::RefCell)
    r = (E - rc.E_ref) / rc.n_atoms - μ  # per-atom energy residual, shifted by the calibrated offset
    # ρ/τ terms are relative squared L2 errors (dimensionless, size-invariant), so dvol cancels.
    (; E=r^2,
       ρ=sum(abs2, ρ .- rc.ρ_ref) / sum(abs2, rc.ρ_ref),
       τ=sum(abs2, τ .- rc.τ_ref) / sum(abs2, rc.τ_ref))
end

function total_loss(config, E, ρ, τ, μ, rc::RefCell)
    l = loss_channels(E, ρ, τ, μ, rc)
    config.loss_weights.E * l.E + config.loss_weights.ρ * l.ρ + config.loss_weights.τ * l.τ
end

function cell_loss(config, rc::RefCell, θ, μ; ρ0=nothing, τ0=nothing, ψ0=nothing)
    if iszero(config.loss_weights.ρ) && iszero(config.loss_weights.τ)
        return cell_loss_hf(config, rc, θ, μ; ρ0, τ0, ψ0)
    end
    basis = build_basis(config, rc, θ)
    warm  = isnothing(ρ0) ? (;) :
            (; ρ=ForwardDiff.value.(ρ0), τ=ForwardDiff.value.(τ0),
               ψ=map(ψk -> ForwardDiff.value.(ψk), ψ0))  # SCF warm-start must be dual-free
    scfres = self_consistent_field(basis; tol=config.tol_scf, warm...)
    total_loss(config, scfres.energies.total, scfres.ρ, scfres.τ, μ, rc), scfres
end

"""Hellmann-Feynman shortcut for the E-only loss: run one primal (dual-free) SCF,
then evaluate only the XC term's energy at a dual basis (θ carries the gradient
direction) with ψ/ρ/τ held fixed at their converged primal values."""
function cell_loss_hf(config, rc::RefCell, θ, μ; ρ0=nothing, τ0=nothing, ψ0=nothing)
    θ_primal = ForwardDiff.value.(θ)
    basis_primal = build_basis(config, rc, θ_primal)
    warm = isnothing(ρ0) ? (;) : (; ρ=ρ0, τ=τ0, ψ=ψ0)  # already dual-free by construction
    scfres = self_consistent_field(basis_primal; tol=config.tol_scf, warm...)

    basis_dual = build_basis(config, rc, θ)
    xc_term_dual = only(filter(t -> t isa DFTK.TermXc, basis_dual.terms))
    T = eltype(basis_dual.model.lattice)
    E_xc_dual = DFTK.energy(xc_term_dual, basis_dual, scfres.ψ, scfres.occupation;
                            ρ=T.(scfres.ρ), τ=T.(scfres.τ), scfres.eigenvalues, scfres.εF)
    E_total_dual = E_xc_dual + (scfres.energies.total - scfres.energies.Xc)

    total_loss(config, E_total_dual, scfres.ρ, scfres.τ, μ, rc), scfres
end

# --- Training ---------------------------------------------------------------------

"""Full-batch loss over the trainset, at flat parameters x = [θ; μ].

`ρ0s[i]`/`τ0s[i]`/`ψ0s[i]` are the warm-start density/kinetic-energy-density/orbitals
for cell i, updated in place as a side effect."""
function make_loss_fn(config, cells, unflatten, ρ0s, τ0s, ψ0s)
    function loss_fn(x::AbstractVector{T}) where {T}
        θ, μ = unflatten(x)
        losses = map(1:length(cells)) do i
            l, scfres = cell_loss(config, cells[i], θ, μ; ρ0=ρ0s[i], τ0=τ0s[i], ψ0=ψ0s[i])
            ρ0s[i] = ForwardDiff.value.(scfres.ρ)
            τ0s[i] = ForwardDiff.value.(scfres.τ)
            ψ0s[i] = map(ψk -> ForwardDiff.value.(ψk), scfres.ψ)
            l
        end
        sum(losses) / length(losses)
    end
end

function calibrate_shift(config, cells, θ0)
    residuals = map(cells) do rc
        basis  = build_basis(config, rc, θ0)
        scfres = self_consistent_field(basis; tol=config.tol_scf)
        (scfres.energies.total - rc.E_ref) / rc.n_atoms
    end
    sum(residuals) / length(residuals)
end

"""Resume from the latest snapshot in workdir if any, else run the initial SCFs
(doubling as warm starts) and calibrate μ. Returns (θ0, μ0, iter0)."""
function restore_or_initialize!(config, workdir, cells)
    snaps = snapshot_files(workdir)
    if isempty(snaps)
        functional0 = MGGAExchange(; config.hidden_sizes)
        θ0 = functional0.parameters
        μ0 = calibrate_shift(config, cells, θ0)
        @printf "calibrated shift μ = %.6f Ha/atom\n" μ0
        (θ0, μ0, 0)
    else
        path = last(snaps)
        iter = snapshot_iter(path)
        @printf "resuming from %s\n" path
        snap = load(path)
        (snap["θ"], snap["μ"], iter)
    end
end

function train(config, workdir, num_train_steps=5)
    mkpath(workdir)
    cells = load_ref_cells(config, [cell_name(phase, v) for (phase, v) in config.trainset])

    θ0, μ0, iter0 = restore_or_initialize!(config, workdir, cells)
    iter0 >= num_train_steps && return (vcat(θ0, μ0), [])  # finished run: no-op

    θ_axes = ComponentArrays.getaxes(θ0)
    unflatten(x) = (ComponentArray(x[1:end-1], θ_axes), x[end])
    x0 = vcat(θ0, μ0)
    ρ0s = Vector{Any}(nothing, length(cells))
    τ0s = Vector{Any}(nothing, length(cells))
    ψ0s = Vector{Any}(nothing, length(cells))
    loss_fn = make_loss_fn(config, cells, unflatten, ρ0s, τ0s, ψ0s)

    trajectory = []
    function callback(os)
        it = iter0 + os.iteration
        θ_it, μ_it = unflatten(os.metadata["x"])
        jldsave(snapshot_path(workdir, it); θ=θ_it, μ=μ_it, iteration=it, loss=os.value, config)
        push!(trajectory, (; iteration=it, os.value, time=time()))
        @printf "bfgs %4d   loss %.6e\n" it os.value
        false
    end

    res = Optim.optimize(loss_fn, x0, BFGS(; initial_stepnorm=0.01, linesearch=BackTracking()),
                         autodiff=:forward,
                         Optim.Options(; iterations=num_train_steps - iter0, g_tol=1e-6,
                                       show_trace=true, extended_trace=true, callback))
    println(res)
    Optim.minimizer(res), trajectory
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) < 2 && error("usage: julia train.jl <config.jl> <workdir> [num_train_steps]")
    include(ARGS[1])
    workdir = ARGS[2]
    num_train_steps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    train(get_config(), workdir, num_train_steps)
end
