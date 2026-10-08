# Evaluation compute for the loss-variant comparison of the paper. Compute only -
# figures.jl renders the figure and the quoted numbers from the files written here.
#
#   julia +1.12.5 -t 8 --project evaluate.jl <rundir> [step,step,...]   # default: last snapshot
#
# Per snapshot, on the diamond and β-tin cells (7 volumes each): the self-consistent
# total energy and the relative L2 density error against HSE06.
# Writes <rundir>/eval/bfgs_<step>.jld2.

using DFTK
using JLD2
using LinearAlgebra: det
using Printf

include("input_pipeline.jl")
include("xc_model.jl")
include("train.jl")

DFTK.setup_threading()

const EVAL_PHASES = ("diamond", "bsn")

function evaluate_snapshot(rundir, step)
    snap   = load(snapshot_path(rundir, step))
    config = snap["config"]
    cells  = Dict{String,Any}()
    for rc in load_ref_cells(config, [cell_name(p, v) for p in EVAL_PHASES for v in VOLUMES])
        scfres = self_consistent_field(build_basis(config, rc, snap["θ"]);
                                       tol=config.tol_scf, callback=identity)
        cells[rc.name] = (; E_ref=rc.E_ref, rc.n_atoms, V_atom=abs(det(rc.lattice)) / rc.n_atoms,
                            E_sc=scfres.energies.total, ρ_err=rel_density_error(scfres.ρ, rc.ρ_ref))
        @printf "step %3d %-12s E_sc %.8f Ha\n" step rc.name scfres.energies.total
    end
    out = joinpath(mkpath(joinpath(rundir, "eval")), @sprintf("bfgs_%04d.jld2", step))
    jldsave(out; step, μ=snap["μ"], config, cells)
    @info "wrote $out"
end

if abspath(PROGRAM_FILE) == @__FILE__
    rundir = abspath(ARGS[1])
    steps  = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) :
             [snapshot_iter(last(snapshot_files(rundir)))]
    foreach(step -> evaluate_snapshot(rundir, step), steps)
end
