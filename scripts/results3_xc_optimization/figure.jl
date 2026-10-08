# The loss-variant figure of the paper and the numbers quoted in its text, rendered
# purely from the files written by evaluate.jl (never runs an SCF).
#
#   julia +1.12.5 --project figures.jl <rundir> <rundir> ...
#
# Uses each run's evaluation of its last snapshot, the initial functional from
# whichever run has eval/bfgs_0000.jld2, and the PBE energies and densities of the
# QE dataset. Writes runs/figures/ev_variants.pdf and the sidecar ev_variants.json
# holding the evaluated checkpoints, the raw per-cell energies and the numbers quoted
# in the text (also printed):
#   * β-tin - diamond energy offset ΔE₀ (Birch-Murnaghan minima) per functional
#   * range of the density errors over the volumes

using CairoMakie
using JLD2
using JSON3

include("input_pipeline.jl")
include("eos.jl")

const EV_PER_HA = 27.211386245988
const Å3_PER_BOHR3 = 0.529177210903^3
to_eV(E)  = E * EV_PER_HA
to_meV(E) = 1000 * to_eV(E)
to_Å3(V)  = V * Å3_PER_BOHR3

# Okabe-Ito colors, redundant linestyles and markers so grayscale print survives
const STYLE = Dict("initial" => ("#E69F00", :solid, :xcross),
                   "PBE"     => ("#999999", :solid,   :rect),
                   "E"       => ("#0072B2", :dot,     :utriangle),
                   "E+ρ"     => ("#009E73", :solid,   :dtriangle),
                   "E+ρ+τ"   => ("#D55E00", :dot, :circle),
                   "target"  => ("#000000", :dash,    :diamond))
const VARIANT_ORDER = ["E", "E+ρ", "E+ρ+τ"]
variant_label(config) = join([String(c) for (c, w) in pairs(config.loss_weights) if w > 0], "+")

"""Evaluation of a snapshot (same file name as the snapshot, under eval/)."""
load_eval(rundir, snapshot) = load(joinpath(rundir, "eval", snapshot))
last_snapshot(rundir) = last(sort(filter(f -> occursin(r"^bfgs_\d{4}\.jld2$", f), readdir(rundir))))

# --- Load ------------------------------------------------------------------------

rundirs = abspath.(ARGS)
variants = map(rundirs) do rundir
    final = load_eval(rundir, last_snapshot(rundir))
    (; rundir, label=variant_label(final["config"]), final)
end
sort!(variants; by=v -> findfirst(==(v.label), VARIANT_ORDER))
initial = let rundir = rundirs[findfirst(d -> isfile(joinpath(d, "eval", "bfgs_0000.jld2")), rundirs)]
    load_eval(rundir, "bfgs_0000.jld2")
end

cells = first(variants).final["cells"]   # cell metadata and E_ref are shared
pbe = let E = reference_energies("pbe"), ρ = reference_densities("pbe"),
          ρ_ref = reference_densities(first(variants).final["config"].ref_functional)
    Dict(n => (; c.n_atoms, E_sc=E[n], ρ_err=rel_density_error(ρ[n], ρ_ref[n])) for (n, c) in cells)
end
phase_of(name) = split(name, "_v")[1]
phase_names(p) = sort([n for n in keys(cells) if phase_of(n) == p], by=n -> cells[n].V_atom)
dia, bsn = phase_names("diamond"), phase_names("bsn")
trained_phases = unique(first.(first(variants).final["config"].trainset))
phase_title(p, name) = name * (p in trained_phases ? " (train)" : " (held out)")

per_atom(cs, key) = n -> getproperty(cs[n], key) / cs[n].n_atoms
e_ref = per_atom(cells, :E_ref)

# (label, per-cell evaluation results, style)
methods = [
    ("initial (LDA-x + PBE-c)", initial["cells"], STYLE["initial"]),
    ("PBE", pbe, STYLE["PBE"]),
    [("loss $(v.label)", v.final["cells"], STYLE[v.label]) for v in variants]...,
]

diamond_volumes = map(0:6) do i
    to_Å3(cells["diamond_v0$i"].V_atom)
end

# --- Figure ------------------------------------------------------------------------

function bm_branch(ns, E)
    Vs = [cells[n].V_atom for n in ns]
    (; fit=eos_birch_murnaghan_fit(Vs, E.(ns)), Vg=range(extrema(Vs)...; length=200))
end
bm_energy(b) = eos_birch_murnaghan_energy.(b.Vg; b.fit...)

function build()
    # these are relative to 1 CSS px
    inch = 96
    pt = 4/3
    cm = inch / 2.54
    default_width = 2 * 8.6 * cm  # Double column width
    fontsize = 10pt

    lw = 2.5
    ms = 12

    fig = Figure(; size=(default_width, 0.52 * default_width), fontsize, figure_padding=2)
    axE = (Axis(fig[1, 1]; title=phase_title("diamond", "diamond"),
                ylabel="E - E₀(diamond) (eV/atom)", xticks=diamond_volumes),
           Axis(fig[1, 2]; title=phase_title("bsn", "β-tin")))
    axρ = (Axis(fig[2, 1]; xlabel="Volume (Å³/atom)", ylabel="Density error (%)",
                yticks=0:2:4, xticks=diamond_volumes, xtickformat = "{:.1f}"),
           Axis(fig[2, 2]; xlabel="Volume (Å³/atom)", yticks=0:2:4))
    for (pos, lab) in zip(((1, 1), (1, 2), (2, 1), (2, 2)), ("(a)", "(b)", "(c)", "(d)"))
        xpad = pos[2] == 1 ? 30 : 0
        Label(fig[pos..., TopLeft()], lab; font=:bold, fontsize, halign=:left,
              padding=(0, xpad, 10, 0))
    end

    curves = [[(label, per_atom(cs, :E_sc), style) for (label, cs, style) in methods];
              ("HSE06 (target)", e_ref, STYLE["target"])]
    for (label, E, (color, linestyle, marker)) in curves
        dia_b = bm_branch(dia, E)
        e0 = dia_b.fit.e0
        for (ax, ns, b) in zip(axE, (dia, bsn), (dia_b, bm_branch(bsn, E)))
            lines!(ax, to_Å3.(b.Vg), to_eV.(bm_energy(b) .- e0); color, linestyle,
                   linewidth=lw, label)
            scatter!(ax, [to_Å3(cells[n].V_atom) for n in ns], to_eV.(E.(ns) .- e0);
                     color)#, marker, markersize=ms)
        end
    end

    for (ax, ns) in zip(axρ, (dia, bsn))
        Vs = [to_Å3(cells[n].V_atom) for n in ns]
        for (_, cs, (color, linestyle, marker)) in methods
            errs = [100 * cs[n].ρ_err for n in ns]
            lines!(ax, Vs, errs; color, linestyle, linewidth=lw)
            scatter!(ax, Vs, errs; color)#, marker, markersize=ms)
        end
    end

    linkyaxes!(axE...); linkyaxes!(axρ...)
    foreach(ax -> ylims!(ax; low=0), axρ)
    for (top, bottom) in zip(axE, axρ)
        linkxaxes!(top, bottom)
        hidexdecorations!(top; grid=false, ticks=false)
    end
    foreach(ax -> hideydecorations!(ax; grid=false, ticks=false), (axE[2], axρ[2]))
    rowsize!(fig.layout, 2, Relative(0.34))
    rowgap!(fig.layout, 1, fontsize / 2)

    Legend(fig[1:2, 3],
           [[LineElement(; color, linestyle, linewidth=lw),
             MarkerElement(; color, marker=:circle)]#, marker, markersize=ms)]
            for (_, _, (color, linestyle, marker)) in curves],
           first.(curves); framevisible=false, rowgap=2)
    fig
end

# --- Numbers quoted in the text ----------------------------------------------------

ΔE0(E) = bm_branch(bsn, E).fit.e0 - bm_branch(dia, E).fit.e0
ΔE0_ref = ΔE0(e_ref)
density_error_range(cs) = (; (Symbol(p) => collect(extrema(100 * cs[n].ρ_err for n in ns))
                              for (p, ns) in (("diamond", dia), ("bsn", bsn)))...)

# JSON objects keep insertion order: functionals as in the legend, cells by name
ordered(kvs) = (; (Symbol(k) => v for (k, v) in kvs)...)
cell_order = [sort(dia); sort(bsn)]

reported = (;
    energy_offset_bsn_minus_diamond_meV_per_atom=(;
        ordered((label, to_meV(ΔE0(per_atom(cs, :E_sc)))) for (label, cs, _) in methods)...,
        var"HSE06 (target)"=to_meV(ΔE0_ref)),
    density_error_percent_min_max=ordered((label, density_error_range(cs)) for (label, cs, _) in methods),
)

raw_cells(cs) = ordered((n, (; cells[n].V_atom, cs[n].n_atoms, cs[n].E_sc, cs[n].ρ_err))
                        for n in cell_order)
raw = (;
    units="energies Ha per cell, V_atom bohr³/atom; E_sc self-consistent, " *
          "ρ_err relative L2 density error vs HSE06",
    ordered((label, raw_cells(cs)) for (label, cs, _) in methods)...,
    var"HSE06 (target)"=ordered((n, (; cells[n].V_atom, cells[n].n_atoms, cells[n].E_ref)) for n in cell_order),
    theta_Si_Ha_per_atom=ordered(("loss $(v.label)", v.final["μ"]) for v in variants),
)

checkpoints = ordered(("loss $(v.label)", joinpath(basename(v.rundir), "bfgs_$(lpad(v.final["step"], 4, '0')).jld2"))
                      for v in variants)

outdir = @__DIR__
with_theme(theme_latexfonts()) do
save(joinpath(outdir, "figure_lossvariants.pdf"), build();
     pt_per_unit=1)
end
open(io -> JSON3.pretty(io, (; checkpoints, reported, raw)), joinpath(outdir, "figure_lossvariants.json"), "w")
JSON3.pretty(stdout, reported)
println()
@info "wrote $(joinpath(outdir, "figure_lossvariants.{pdf,json}"))"
