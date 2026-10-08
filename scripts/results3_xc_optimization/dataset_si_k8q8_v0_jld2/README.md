# Si (ρ, τ) dataset - k8q8 v0 (JLD2)

One JLD2 file per XC functional (PBE, HSE06); each holds 35 cells
(5 phases × 7 volumes), keyed `"cells/<phase>_v<NN>"`. List them with
`jldopen("hse06.jld2", "r") do io; keys(io["cells"]); end`.
Each cell is a NamedTuple: `ρ` (e/bohr³) and `τ` (Ha/bohr³, Hartree atomic units) on the FFT
grid in DFTK order `(n1,n2,n3,nspin)`, plus lattice/k-mesh/ecut metadata and verbatim `scf_in`/`scf_out`.

Setup: silicon, norm-conserving SG15 (no NLCC), 8×8×8 k- and q-mesh, ecutwfc 50 Ry.

## Loading the data

```julia
using JLD2
cell = load("hse06.jld2", "cells/diamond_v03")
cell.ρ  # array shape (n1,n2,n3,nspin)
cell.τ  # array shape (n1,n2,n3,nspin)
```
See `example_load.jl` for a runnable end-to-end demo.
