using DFTK
using PseudoPotentialData

pseudopotentials = Dict(
    :Ba => "pseudos/Ba_v1.upf",
    :Ti => "pseudos/Ti_v1.upf",
    :O => "pseudos/O_v2.upf",
)
model_kwargs = (; functionals=[:mgga_x_r2scan01, :mgga_c_r2scan01], pseudopotentials,
                  temperature=1e-3, smearing=Smearing.FermiDirac(),
                  kinetic_blowup=BlowupCHV())
basis_kwargs = (; kgrid=[8, 8, 8], Ecut=60)
calc = DFTKCalculator(; model_kwargs, basis_kwargs)