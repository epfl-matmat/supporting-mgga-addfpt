#!/bin/sh
## To be called like `bash <filename>.jl`.
#=
BN=$(basename "$0" .jl)
~/.julia/bin/mpiexecjl --project -n 1 julia +1.12.5 -t8 --heap-size-hint 80G $BN.jl >${BN}.log 2>&1
exit $?
=#
using AtomsBase
using AtomsIO
using GeometryOptimization
using LinearAlgebra
using Unitful
using UnitfulAtomic

include("0_params.jl")

setup_threading()

initial_system = load_system("structures/ZnO.extxyz")

tol_forces = austrip(1e-4u"eV/Å")
tol_stress = austrip(0.1u"kbar")
initial_volume = austrip(det(stack(cell_vectors(initial_system))))
tol_virial = initial_volume * tol_stress
println("Relaxing with tol_forces = $tol_forces and tol_virial = $tol_virial (in a.u.)...")
results = minimize_energy!(initial_system, calc; variablecell=true,
                           tol_forces, tol_virial, verbosity=2)

save_system("structures/ZnO_relaxed.extxyz", results.system)