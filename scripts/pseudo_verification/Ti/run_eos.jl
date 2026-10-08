#!/bin/sh
## To be called like `bash <filename>.jl <structure> <Ecut> <version>`.
#=
BN=$(basename "$0" .jl)
STRUCTURE=$1
Ecut=$2
VERS=$3
~/.julia/bin/mpiexecjl --project -n 8 julia +1.12.5 --heap-size-hint 10G ${BN}.jl $STRUCTURE $Ecut $VERS >${BN}_${STRUCTURE}_${Ecut}_v${VERS}.log 2>&1
exit $?
=#
using AtomsIO
using DFTK
using JSON3
using OrderedCollections
using Unitful
using UnitfulAtomic

setup_threading()

STRUCTURE = ARGS[1]
Ecut = parse(Float64, ARGS[2])
VERS = parse(Int, ARGS[3])
system = load_system("$STRUCTURE.xsf")
psp = load_psp("Ti_v$VERS.upf")
pseudopotentials = fill(psp, length(system))
model0 = model_DFT(system; functionals = [:mgga_x_r2scan01, :mgga_c_r2scan01],
                   pseudopotentials, temperature=0.0045/2, smearing=Smearing.FermiDirac(),
                   kinetic_blowup=BlowupCHV())

volumes = Float64[]
energies = Float64[]

consistent_kgrid = nothing

for scales in 0.94:0.02:1.06
    lattice = cbrt(scales) * model0.lattice
    model = Model(model0; lattice)

    if isnothing(consistent_kgrid)
        basis = PlaneWaveBasis(model; Ecut, kgrid=KgridSpacing(0.12u"Å^-1"))
        global consistent_kgrid = basis.kgrid
    else
        basis = PlaneWaveBasis(model; Ecut, kgrid=consistent_kgrid)
    end

    scfres = self_consistent_field(basis; tol=1e-5, mixing=SimpleMixing())

    push!(volumes, model.unit_cell_volume)
    push!(energies, scfres.energies.total)
end

if mpi_master()
    data = OrderedDict(
        :volumes => volumes,
        :energies => energies,
    )

    open("data_$(STRUCTURE)_$(Ecut)_v$(VERS).json", "w") do f
        JSON3.pretty(f, data)
    end
end
