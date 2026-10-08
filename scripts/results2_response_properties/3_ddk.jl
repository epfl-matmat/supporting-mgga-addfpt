#!/bin/sh
## To be called like `bash <filename>.jl`.
#=
BN=$(basename "$0" .jl)
~/.julia/bin/mpiexecjl --project -n 1 julia +1.12.5 -t8 --heap-size-hint 80G $0 >${BN}.log 2>&1
exit $?
=#
using AtomsIO
using JLD2

WD = pwd()
include("$WD/0_params.jl")

setup_threading()

println("Loading SCF results...")
scfres = load_scfres("$WD/outputs/scfres.jld2")
println("Loaded SCF results.")

bandtolalg = DFTK.BandtolBalanced(scfres.ham.basis, scfres.ψ, scfres.occupation; scfres.occupation_threshold)
δψ_dk = DFTK.compute_ψbars(scfres.ham, scfres.ψ, scfres.occupation,
                           scfres.εF, scfres.eigenvalues; scfres.ρ, scfres.τ,
                           scfres.occupation_threshold, bandtolalg,
                           tol=1e-9);
δψ_dk = [x.ψbar for x in δψ_dk]
jldopen("$WD/outputs/ddk.jld2", "w") do f
    f["δψ_dk"] = δψ_dk
end

println("All finished!")
