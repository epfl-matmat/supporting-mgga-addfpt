# JLD2 dataset bundle (Quantum ESPRESSO reference calculations) -> DFTK cell data,
# pinning the basis to the QE numerics: lattice, embedded pseudopotentials, FFT grid,
# k-grid, Ecut and smearing. All quantities in Hartree atomic units.

using JLD2
using DFTK
using LinearAlgebra

const DATASET_DIR = joinpath(@__DIR__, "dataset_si_k8q8_v0_jld2")
const FUNCTIONALS = ("pbe", "hse06")
const PHASES      = ("diamond", "bsn", "hex", "fcc", "imma")
const VOLUMES     = 0:6

bundle_path(functional) = joinpath(DATASET_DIR, "$functional.jld2")
cell_name(phase, volume) = "$(phase)_v$(lpad(volume, 2, '0'))"

load_cell(bundle, name) = load(bundle, "cells/$name")

"""Materialize the embedded .upf bytes as temp files and load them, once per bundle."""
function load_pseudopotentials(bundle)
    dir = mktempdir()
    Dict(map(collect(load(bundle, "pseudopotentials"))) do (name, bytes)
        path = joinpath(dir, name)
        write(path, bytes)
        name => load_psp(path)
    end)
end

function dftk_structure(cell, psps)
    (; lattice, atoms) = cell.atomic_structure
    positions = [lattice \ a.position for a in atoms]  # QE stores Cartesian bohr
    elements  = [ElementPsp(Symbol(a.name); psp=psps[cell.atomic_species[a.name].pseudo_file])
                 for a in atoms]
    lattice, elements, positions
end

function parse_smearing(cell)
    cell.smearing.type == "gaussian" || error("Unsupported smearing: $(cell.smearing.type)")
    (; temperature=cell.smearing.degauss_Ha, smearing=Smearing.Gaussian())
end

"""Relative L2 density error ‖ρ - ρ_ref‖ / ‖ρ_ref‖ over the cell grid (dvol cancels)."""
rel_density_error(ρ, ρ_ref) = norm(ρ - ρ_ref) / norm(ρ_ref)

reference_energies(functional) = jldopen(bundle_path(functional), "r") do f
    Dict(n => f["cells/$n"].qe_energy_terms_Ha.etot for n in keys(f["cells"]))
end

reference_densities(functional) = jldopen(bundle_path(functional), "r") do f
    Dict(n => f["cells/$n"].ρ for n in keys(f["cells"]))
end
