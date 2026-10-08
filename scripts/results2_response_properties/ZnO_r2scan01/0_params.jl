using DFTK
using PseudoPotentialData

pseudopotentials = Dict(
    :Zn => "pseudos/Zn_v4.upf",
    :O => "pseudos/O_v2.upf",
)
model_kwargs = (; functionals=[:mgga_x_r2scan01, :mgga_c_r2scan01], pseudopotentials,
                  temperature=1e-3, smearing=Smearing.FermiDirac(),
                  kinetic_blowup=BlowupCHV())
basis_kwargs = (; kgrid=[12, 12, 8], Ecut=60)
calc = DFTKCalculator(; model_kwargs, basis_kwargs)