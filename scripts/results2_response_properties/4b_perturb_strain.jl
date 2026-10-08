#!/bin/sh
## To be called like `bash <filename>.jl <index>`.
#=
BN=$(basename "$0" .jl)
IDX=$1
~/.julia/bin/mpiexecjl --project -n 1 julia +1.12.5 -t8 --heap-size-hint 80G $0 $IDX >${BN}_${IDX}.log 2>&1
exit $?
=#
using AtomsIO
using ComponentArrays
using ForwardDiff
using JLD2

WD = pwd()
include("$WD/0_params.jl")

IDX=parse(Int, ARGS[1])
@assert 1 <= IDX <= 6
println("Running with index $IDX")

setup_threading()

println("Loading SCF results...")
scfres = load_scfres("$WD/outputs/scfres.jld2")
basis = scfres.basis
println("Loaded SCF results.")

println("Loading δψ_dk...")
@load "$WD/outputs/ddk.jld2" δψ_dk
println("Loaded δψ_dk.")

function make_strained_basis(η)
    strained_lattice = DFTK.voigt_strain_to_full(η) * basis.model.lattice
    strained_model = Model(basis.model; lattice=strained_lattice, symmetries=false)
    PlaneWaveBasis(basis; model=strained_model)
end

println("Computing δHψ...")
function Hψ(η)
    strained_basis = make_strained_basis(η)
    strained_ρ = compute_density(strained_basis, scfres.ψ, scfres.occupation)
    strained_τ = isnothing(scfres.τ) ? nothing : compute_kinetic_energy_density(strained_basis, scfres.ψ, scfres.occupation)
    strained_ham = energy_hamiltonian(strained_basis, scfres.ψ, scfres.occupation;
                                      ρ=strained_ρ, τ=strained_τ, scfres.eigenvalues, scfres.εF).ham
    strained_ham * scfres.ψ
end
dη = zeros(6)
dη[IDX] = 1.0

δHψ = ForwardDiff.derivative(ε -> Hψ(ε * dη), 0.0);

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
    strained_basis = make_strained_basis(ε * dη)
    ψ = scfres.ψ .+ ε .* dfpt_res.δψ
    ρ = compute_density(strained_basis, ψ, scfres.occupation)
    τ = isnothing(scfres.τ) ? nothing : compute_kinetic_energy_density(strained_basis, ψ, scfres.occupation)
    F = compute_forces_cart(strained_basis, ψ, scfres.occupation; ρ, τ)
    stress = compute_stresses_cart(strained_basis, ψ, scfres.occupation; scfres.eigenvalues, scfres.εF)
    ComponentArray(; F, stress)
end
dq = ForwardDiff.derivative(force_and_stress, 0.0)

jldopen("$WD/outputs/perturb_strain_$IDX.jld2", "w") do f
    f["dP"] = dP
    f["dF"] = stack(dq.F)
    f["dstress"] = Matrix(dq.stress)
end

println("All finished!")
