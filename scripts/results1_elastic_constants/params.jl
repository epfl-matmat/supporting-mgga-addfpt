using DFTK
using Unitful
using UnitfulAtomic

pseudopotentials = Dict(
    :C => "C_v4.upf",
)
model_kwargs = (; functionals = [:mgga_x_r2scan01, :mgga_c_r2scan01], pseudopotentials,
                  temperature=1e-3, smearing=Smearing.FermiDirac(),
                  kinetic_blowup=BlowupCHV())

scf_tol = 1e-8
scf_kwargs = (; tol=scf_tol, mixing=SimpleMixing())
