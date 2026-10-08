using DFTK
using PseudoPotentialData

pseudopotentials = PseudoFamily("dojo.nc.sr.pbe.v0_5.standard.upf")
model_kwargs = (; functionals=PBE(), pseudopotentials,
                  temperature=1e-3, smearing=Smearing.FermiDirac(),
                  kinetic_blowup=BlowupCHV())
basis_kwargs = (; kgrid=[12, 12, 8], Ecut=48)
calc = DFTKCalculator(; model_kwargs, basis_kwargs)