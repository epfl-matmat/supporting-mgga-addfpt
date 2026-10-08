# Minimal end-to-end example for the Si (ρ, τ) JLD2 bundle.
# Run: julia +1.12.5 --project example_load.jl
#
# Demonstrates: opening a per-functional file, reading the file-level metadata,
# loading a single cell, and verifying the two basic physical integrals -
# ∫ρ d³r = number of electrons, and ∫τ d³r = the kinetic energy (Hartree).
using JLD2
using LinearAlgebra: det

# Adjust BUNDLE / CELL to whatever you want to inspect. Files are per-functional
# (pbe / hse06); cell keys are <phase>_v<NN> with phase
# in {diamond, bsn, hex, fcc, imma} and NN in 00..06.
BUNDLE = joinpath(@__DIR__, "hse06.jld2")
CELL   = "diamond_v03"

# --- 1. Open the file, pull metadata + one cell + the pseudopotentials Dict ---
# Top-level layout: file["meta"], file["pseudopotentials"], file["cells/<name>"].
meta, cell, pseudos = jldopen(BUNDLE, "r") do io
    io["meta"], io["cells/$CELL"], io["pseudopotentials"]
end

println("Bundle  : $(meta.functional)  system=$(meta.system)  schema=$(meta.schema_version)")
println("Cell    : $CELL  ($(cell.phase) phase, volume index $(cell.volume_index))")
println("Built   : $(meta.provenance.built)  commit=$(meta.provenance.describe)")
println("Pseudos : ", collect(keys(pseudos)))
println()

# --- 2. Grid geometry (Hartree atomic units throughout: ρ in e/bohr³, τ in Ha/bohr³).
n1, n2, n3 = cell.fft_size
Ω  = abs(det(cell.atomic_structure.lattice))   # cell volume in bohr³
dV = Ω / (n1 * n2 * n3)                         # real-space voxel in bohr³

# Cell geometry, basis / FFT grid.
println("lattice (bohr, one vector per line):")
for v in eachcol(cell.atomic_structure.lattice)
    println("  ", v)
end
println("cell volume       = $(round(Ω, digits=4)) bohr³")
println("fft_size          = $(cell.fft_size)   (real-space grid)")
println("array shapes      : ρ $(size(cell.ρ)), τ $(size(cell.τ))   (n1,n2,n3,nspin)")
println("ecutwfc / ecutrho = $(cell.ecutwfc) Ha / $(cell.ecutrho) Ha")
println()

# --- 3. Density integrates to N_electrons. ---
# ρ has shape (n1,n2,n3,nspin); for closed-shell Si nspin=1.
ρ = cell.ρ[:, :, :, 1]
nelec_integrated = sum(ρ) * dV
@assert isapprox(nelec_integrated, cell.nelec; rtol=1e-6) "∫ρ d³r ≠ nelec"
println("∫ρ d³r           = $(round(nelec_integrated, digits=6)) e   (expected nelec = $(cell.nelec))")

# --- 4. Kinetic-energy density integrates to the kinetic energy. ---
# τ = ½ Σ_i f_i |∇ψ_i|² in Hartree atomic units (the bundle pre-converts from
# QE's Rydberg-style raw ekin-density.dat), so ∫τ d³r is literally the kinetic
# energy T in Ha. T is independently a positive contribution to E_total of
# order ~1 Ha/atom for Si.
τ = cell.τ[:, :, :, 1]
T_kin = sum(τ) * dV
natoms = length(cell.atomic_structure.atoms)
println("∫τ d³r (= T_kin) = $(round(T_kin, digits=6)) Ha  (= $(round(T_kin/natoms, digits=3)) Ha/atom)")
@assert T_kin > 0 && 0.1 < T_kin / cell.nelec < 100 "T_kin not in the expected range"

# --- 5. QE's energy decomposition is in cell.qe_energy_terms_Ha (Hartree). ---
println()
println("QE energy decomposition (Hartree):")
for (k, v) in pairs(cell.qe_energy_terms_Ha)
    println("  $(rpad(string(k), 6)) = ", v)
end

# --- 6. Enumerate all cells in this file.
println()
all_cells = jldopen(BUNDLE, "r") do io; sort(collect(keys(io["cells"]))); end
println("All $(length(all_cells)) cells in $(basename(BUNDLE)): ", first(all_cells, 4), " … ", last(all_cells, 2))
