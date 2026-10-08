#!/bin/sh
## To be called like `bash <filename>.jl <structure>`.
#=
BN=$(basename "$0" .jl)
STRUCTURE=$1
julia +1.12.5 --project -t4 --heap-size-hint 20G $BN.jl $STRUCTURE >${BN}_${STRUCTURE}.log 2>&1
exit $?
=#
using AtomsBase
using AtomsIO
using GeometryOptimization
using LinearAlgebra
using Unitful
using UnitfulAtomic
include("params.jl")

STRUCTURE = ARGS[1]

if STRUCTURE == "C"
    Ecut = 55
    kgrid = [8, 8, 8]
elseif STRUCTURE == "Si"
    error("The parameters for structure $STRUCTURE are not known.")
end

setup_threading()

system = load_system("structures/$STRUCTURE.extxyz")
calc = DFTKCalculator(; model_kwargs,
                        basis_kwargs = (; Ecut, kgrid),
                        scf_kwargs)

tol_forces = austrip(1e-4u"eV/Å")
tol_stress = austrip(0.1u"kbar")
initial_volume = austrip(det(stack(cell_vectors(system))))
tol_virial = initial_volume * tol_stress
println("Relaxing with tol_forces = $tol_forces and tol_virial = $tol_virial (in a.u.)...")
results = minimize_energy!(system, calc; variablecell=true,
                           tol_forces, tol_virial, verbosity=2)

save_system("structures/$(STRUCTURE)_relaxed.extxyz", results.system)
