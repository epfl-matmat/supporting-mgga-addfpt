using CairoMakie
using LaTeXStrings
using LinearAlgebra
using Printf

# C11, C12, C44
AD_RESULT  = [1110.6091110617951, 114.40446084149053, 584.5543193083424]

FD_RESULTS = [
    1e-2  => [1110.7143287408971, 114.74507734887484, 584.6570634049363],
    1e-3  => [1110.610080873793, 114.40783556569893, 584.5552205406635],
    1e-4  => [1110.6091530149786, 114.40451064266327, 584.5542454685567],
    1e-5  => [1110.6091401401322, 114.4044782428884, 584.554237124252],
    1e-6  => [1110.609166554913, 114.40448909218799, 584.5542350334939],
    1e-7  => [1110.609693900424, 114.40520444571209, 584.553763160327],
]

# these are relative to 1 CSS px
inch = 96
pt = 4/3
cm = inch / 2.54
default_width = 8.5 * cm  # Single column width
fontsize = 10pt

set_theme!(theme_latexfonts())
fig = Figure(; size=(default_width, 0.6 * default_width), fontsize, figure_padding=2)
ax = Makie.Axis(fig[1,1],
                xlabel=L"FD step $h$",
                ylabel=L"$|C_\mathrm{AD-DFPT}-C_\mathrm{FD}|$ (GPa)",
                xscale=log10,
                yscale=log10)
ax.xreversed = true

C_ELEMENTS = [L"C_{11}", L"C_{12}", L"C_{44}"]
styles = [:circle, :utriangle, :xcross]
for C_idx in 1:3
    hs = first.(FD_RESULTS)
    errors = [abs.(fd_result - AD_RESULT)[C_idx] for (_, fd_result) in FD_RESULTS]
    scatterlines!(ax, hs, errors; label=C_ELEMENTS[C_idx],
                  linewidth=2.5, marker=styles[C_idx], markersize=15)
end
Makie.Legend(fig[1,2], ax)

save("4_findiff_convergence.pdf", fig)

lines = ["AD-DFPT", raw"FD $h = 10^{-5}$"]
for i in 1:3
    ad_C = @sprintf "%.5f" AD_RESULT[i]
    fd_data = FD_RESULTS[4]
    @assert fd_data[1] == 1e-5
    fd_C = @sprintf "%.5f" fd_data[2][i]

    common_digits = findlast(collect(ad_C) .== collect(fd_C))

    lines[1] = lines[1] * " & \\textbf{$(ad_C[1:common_digits])}$(ad_C[common_digits+1:end])"
    lines[2] = lines[2] * " & \\textbf{$(fd_C[1:common_digits])}$(fd_C[common_digits+1:end])"
end
println(lines[1], " \\\\")
println(lines[2], " \\\\")