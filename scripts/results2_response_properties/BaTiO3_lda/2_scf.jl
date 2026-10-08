#!/bin/sh
## To be called like `bash <filename>.jl`.
#=
BN=$(basename "$0" .jl)
~/.julia/bin/mpiexecjl --project -n 1 julia +1.12.5 -t8 --heap-size-hint 80G $BN.jl >${BN}.log 2>&1
exit $?
=#
using AtomsIO
using JLD2

include("0_params.jl")

setup_threading()

system = load_system("structures/BaTiO3_relaxed.extxyz")
model = model_DFT(system; model_kwargs..., symmetries=false)
basis = PlaneWaveBasis(model; basis_kwargs...);

scfres = self_consistent_field(basis; tol=1e-8);

save_scfres("outputs/scfres.jld2", scfres; save_ψ=true, save_ρ=true)

println("All finished!")
