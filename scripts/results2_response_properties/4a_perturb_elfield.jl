#!/bin/sh
## To be called like `bash <filename>.jl <index>`.
#=
BN=$(basename "$0" .jl)
IDX=$1
~/.julia/bin/mpiexecjl --project -n 1 julia +1.12.5 -t8 --heap-size-hint 80G $0 $IDX >${BN}_${IDX}.log 2>&1
exit $?
=#
using AtomsIO
using ForwardDiff
using JLD2

WD = pwd()
include("$WD/0_params.jl")

IDX=parse(Int, ARGS[1])
@assert 1 <= IDX <= 3
println("Running with index $IDX")

setup_threading()

println("Loading SCF results...")
scfres = load_scfres("$WD/outputs/scfres.jld2")
basis = scfres.basis
println("Loaded SCF results.")

println("Loading δψ_dk...")
@load "$WD/outputs/ddk.jld2" δψ_dk
println("Loaded δψ_dk.")

println("Running DFPT computation...")
dfpt_res = DFTK.solve_ΩplusK_split(scfres.ham, scfres.ρ, scfres.ψ,
                                   scfres.occupation, scfres.εF, scfres.eigenvalues,
                                   δψ_dk[IDX];
                                   scfres.τ, tol=1e-8, scfres.occupation_threshold)
println("Finished DFPT computation.")

println("Computing polarization change")
Ω = basis.model.unit_cell_volume
dP = map(1:3) do α
    DFTK.weighted_ksum(basis, map(δψ_dk[α], dfpt_res.δψ, scfres.occupation) do ψkbar, δψk, occk
        -2/Ω * sum(occk .* real.(DFTK.columnwise_dots(ψkbar, δψk)))
    end)
end

println("Computing forces change")
dF = ForwardDiff.derivative(ε -> compute_forces_cart(basis, scfres.ψ .+ ε .* dfpt_res.δψ,
                                                     scfres.occupation;
                                                     ρ=scfres.ρ .+ ε .* dfpt_res.δρ,
                                                     τ=isnothing(scfres.τ) ? nothing : scfres.τ .+ ε .* dfpt_res.δτ),
                            0.0)

println("Computing stress change")
dstress = ForwardDiff.derivative(ε -> compute_stresses_cart(basis, scfres.ψ .+ ε .* dfpt_res.δψ,
                                                            scfres.occupation; scfres.eigenvalues, scfres.εF),
                                 0.0)

jldopen("$WD/outputs/perturb_elfield_$IDX.jld2", "w") do f
    f["dP"] = dP
    f["dF"] = stack(dF)
    f["dstress"] = Matrix(dstress)
end

println("All finished!")
