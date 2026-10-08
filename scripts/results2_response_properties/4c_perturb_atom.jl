#!/bin/sh
## To be called like `bash <filename>.jl <index>`.
#=
BN=$(basename "$0" .jl)
IDX_ATOM=$1
IDX_DIR=$2
~/.julia/bin/mpiexecjl --project -n 1 julia +1.12.5 -t8 --heap-size-hint 80G $0 $IDX_ATOM $IDX_DIR >${BN}_${IDX_ATOM}_${IDX_DIR}.log 2>&1
exit $?
=#
using AtomsIO
using ComponentArrays
using ForwardDiff
using JLD2

WD = pwd()
include("$WD/0_params.jl")

IDX_ATOM=parse(Int, ARGS[1])
IDX_DIR=parse(Int, ARGS[2])
@assert 1 <= IDX_ATOM <= 5
@assert 1 <= IDX_DIR <= 3
println("Running with atom index $IDX_ATOM and direction index $IDX_DIR")

setup_threading()

println("Loading SCF results...")
scfres = load_scfres("$WD/outputs/scfres.jld2")
basis = scfres.basis
println("Loaded SCF results.")

println("Loading δψ_dk...")
@load "$WD/outputs/ddk.jld2" δψ_dk
println("Loaded δψ_dk.")

positions = basis.model.positions
dpositions = zero(positions)
dpositions[IDX_ATOM] = [α == IDX_DIR for α in 1:3]

function make_perturbed_basis(ε)
    perturbed_positions = positions .+ ε .* dpositions
    perturbed_lattice = basis.model.lattice .* (1 + ε * 0)
    perturbed_model = Model(basis.model; lattice=perturbed_lattice, positions=perturbed_positions, symmetries=false)
    PlaneWaveBasis(basis; model=perturbed_model)
end

println("Computing δHψ...")
function Hψ(ε)
    perturbed_basis = make_perturbed_basis(ε)
    # TODO: seems necessary else the Hartree term is unhappy?
    perturbed_ρ = scfres.ρ .* (1 + ε * 0)
    # doing it for τ as well, might not be necessary
    perturbed_τ = isnothing(scfres.τ) ? nothing : scfres.τ .* (1 + ε * 0)
    perturbed_ham = energy_hamiltonian(perturbed_basis, scfres.ψ, scfres.occupation;
                                       ρ=perturbed_ρ, τ=perturbed_τ, scfres.eigenvalues, scfres.εF).ham
    perturbed_ham * scfres.ψ
end

δHψ = ForwardDiff.derivative(Hψ, 0.0);

println("Running DFPT computation...")
dfpt_res = DFTK.solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ,
                                   scfres.occupation, scfres.εF, scfres.eigenvalues,
                                   δHψ;
                                   scfres.τ, tol=1e-8, scfres.occupation_threshold)
println("Finished DFPT computation.")

println("Computing polarization change")
Ω = basis.model.unit_cell_volume
dP = map(1:3) do α
    DFTK.weighted_ksum(basis, map(δψ_dk[α], dfpt_res.δψ, scfres.occupation) do ψkbar, δψk, occk
        -2/Ω * sum(occk .* real.(DFTK.columnwise_dots(ψkbar, δψk)))
    end)
end

println("Computing forces and stress changes")
function force_and_stress(ε)
    perturbed_basis = make_perturbed_basis(ε)
    ψ = scfres.ψ .+ ε .* dfpt_res.δψ
    ρ = compute_density(perturbed_basis, ψ, scfres.occupation)
    τ = isnothing(scfres.τ) ? nothing : compute_kinetic_energy_density(perturbed_basis, ψ, scfres.occupation)
    F = compute_forces_cart(perturbed_basis, ψ, scfres.occupation; ρ, τ)
    stress = compute_stresses_cart(perturbed_basis, ψ, scfres.occupation; scfres.eigenvalues, scfres.εF)
    ComponentArray(; F, stress)
end
dq = ForwardDiff.derivative(force_and_stress, 0.0)

jldopen("$WD/outputs/perturb_atom_$(IDX_ATOM)_$(IDX_DIR).jld2", "w") do f
    f["dP"] = dP
    f["dF"] = stack(dq.F)
    f["dstress"] = Matrix(dq.stress)
end

println("All finished!")
