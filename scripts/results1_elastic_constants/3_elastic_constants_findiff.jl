#!/bin/sh
## To be called like `bash <filename>.jl <structure>`.
#=
BN=$(basename "$0" .jl)
STRUCTURE=$1
~/.julia/bin/mpiexecjl --project -n 4 julia +1.12.5 --heap-size-hint 10G $BN.jl $STRUCTURE >${BN}_${STRUCTURE}.log 2>&1
exit $?
=#
using AtomsIO
using Dates
using JSON3
using OrderedCollections
include("params.jl")
# Use a tighter tolerance!
scf_tol = 1e-12
scf_kwargs = (; tol=scf_tol, mixing=SimpleMixing())

STRUCTURE = ARGS[1]

if STRUCTURE == "C"
    Ecut = 55
    kgrid = [8, 8, 8]
else
    error("The parameters for structure $STRUCTURE are not known.")
end

setup_threading()

system = load_system("structures/$(STRUCTURE)_relaxed.extxyz")

model0 = model_DFT(system; model_kwargs...)

function symmetries_from_strain(model0, voigt_strain)
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries=true)
    model.symmetries
end

function stress_from_strain(voigt_strain; symmetries)
    lattice = DFTK.voigt_strain_to_full(voigt_strain) * model0.lattice
    model = Model(model0; lattice, symmetries)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; scf_kwargs...)
    DFTK.full_stress_to_voigt(compute_stresses_cart(scfres))
end

strain0 = zeros(6)  # The base point is zero strain relative to the relaxed structure
strain_pattern = [1., 0., 0., 1., 0., 0.]  # should yield [c11, c12, c12, c44, 0, 0]

# For elastic constants beyond the bulk modulus, symmetry-breaking strains
# are required. That is, the symmetry group of the crystal is reduced.
# Here we simply precompute the relevant subgroup by applying the automatic
# symmetry detection (spglib) to a finitely perturbed crystal.
symmetries_strain = symmetries_from_strain(model0, 0.01 * strain_pattern)

mpi_master() && @info now() length(model0.symmetries) length(symmetries_strain) strain_pattern

f(voigt_strain) = stress_from_strain(voigt_strain; symmetries=symmetries_strain)

for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    mpi_master() && println("Computing for h=$h")
    stress_ph = f(strain0 + h * strain_pattern)
    stress_mh = f(strain0 - h * strain_pattern)
    dstress_h = (stress_ph - stress_mh) / 2h

    c11_h = ustrip(uconvert(u"GPa", dstress_h[1] * u"hartree" / u"bohr"^3))
    c12_h = ustrip(uconvert(u"GPa", dstress_h[2] * u"hartree" / u"bohr"^3))
    c44_h = ustrip(uconvert(u"GPa", dstress_h[4] * u"hartree" / u"bohr"^3))

    if mpi_master()
        println("h: ", h)
        println("scf_tol: ", scf_tol)
        println("Ecut: ", Ecut)
        println("C11: ", c11_h)
        println("C12: ", c12_h)
        println("C44: ", c44_h)
        println("all three: [", c11_h, ", ", c12_h, ", ", c44_h, "]")
        println("stress_mh: ", stress_mh)
        println("stress_ph: ", stress_ph)
        println("dstress: ", dstress_h)
    end
end
