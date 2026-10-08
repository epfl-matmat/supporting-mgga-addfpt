#!/bin/sh
## To be called like `bash <filename>.jl <system name>`.
#=
BN=$(basename "$0" .jl)
SYSTEM_NAME=$1
julia +1.12.5 --project $0 $SYSTEM_NAME >${BN}.log 2>&1
exit $?
=#
using AtomsIO
using JLD2
using LinearAlgebra
using Printf
using Unitful
using UnitfulAtomic

wd = pwd()
SYSTEM_NAME = ARGS[1]

include(joinpath(wd, "0_params.jl"))

setup_threading()

system = load_system("structures/$(SYSTEM_NAME)_relaxed.extxyz")
model = model_DFT(system; model_kwargs..., symmetries=false)
natoms = length(model.positions)

# Raw perturbations wrt electric field
χ_red = zeros(3, 3)
Zstar1_red = zeros(3, 3, natoms)
e1_red = zeros(3, 6)
for α in 1:3
    @load "outputs/perturb_elfield_$α.jld2" dP dF dstress
    χ_red[α, :] = dP
    Zstar1_red[α, :, :] .= stack(dF)
    e1_red[α, :] = -DFTK.full_stress_to_voigt(dstress)
end

# Raw perturbations wrt strain
C_clamped = zeros(6, 6)
e2_red = zeros(3, 6)
Λ1 = zeros(3, natoms, 6)
for i in 1:6
    @load "outputs/perturb_strain_$i.jld2" dP dF dstress
    C_clamped[i, :] = DFTK.full_stress_to_voigt(dstress)
    e2_red[:, i] = dP
    Λ1[:, :, i] .= stack(dF)
end

# Raw perturbations wrt atom displacements
Zstar2_red_red = zeros(3, 3, natoms)
Λ2_red = zeros(3, natoms, 6)
K_red = zeros(3, natoms, 3, natoms)
for s in 1:natoms, α in 1:3
    @load "outputs/perturb_atom_$(s)_$(α).jld2" dP dF dstress
    Zstar2_red_red[:, α, s] = model.unit_cell_volume * dP
    Λ2_red[α, s, :] = -model.unit_cell_volume * DFTK.full_stress_to_voigt(dstress)
    K_red[α, s, :, :] .= -stack(dF)
end

# Convert atom displacements to Cartesian coordinates
Zstar2_red = zeros(3, 3, natoms)
for s in 1:natoms, α in 1:3
    Zstar2_red[α, :, s] = model.inv_lattice' * Zstar2_red_red[α, :, s]
end
Λ2 = zeros(3, natoms, 6)
for s in 1:natoms, i in 1:6
    Λ2[:, s, i] = model.inv_lattice' * Λ2_red[:, s, i]
end
K = zeros(3, natoms, 3, natoms)
for s in 1:natoms, β in 1:3, t in 1:natoms
    K[:, s, β, t] = model.inv_lattice' * K_red[:, s, β, t]
end

# Convert electric field perturbations to Cartesian coordinates
χ_clamped_cart = model.lattice * χ_red * model.lattice' / (2π)^2
ϵ_clamped_cart = I + 4π * χ_clamped_cart

Zstar1_cart = zeros(3, 3, natoms)
for s in 1:natoms, β in 1:3
    Zstar1_cart[:, β, s] = 1/(2π) * model.lattice * Zstar1_red[:, β, s]
end
for s in 1:natoms
    Zstar1_cart[:, :, s] += I * charge_ionic(model.atoms[s])
end
Zstar2_cart = zeros(3, 3, natoms)
for s in 1:natoms, β in 1:3
    Zstar2_cart[:, β, s] = 1/(2π) * model.lattice * Zstar2_red[:, β, s]
end
for s in 1:natoms
    Zstar2_cart[:, :, s] += I * charge_ionic(model.atoms[s])
end

e1_clamped_cart = 1/(2π) * model.lattice * e1_red
e2_clamped_cart = 1/(2π) * model.lattice * e2_red

# ASR for K
for s in 1:natoms, α in 1:3, β in 1:3
    K[α, s, β, s] -= sum(K[α, s, β, :])
end
# ASR for Zstar
for α in 1:3, β in 1:3
    Zstar1_cart[α, β, :] .-= sum(Zstar1_cart[α, β, :]) / natoms
    Zstar2_cart[α, β, :] .-= sum(Zstar2_cart[α, β, :]) / natoms
end
# ASR for Λ
for α in 1:3, i in 1:6
    Λ1[α, :, i] .-= sum(Λ1[α, :, i]) / natoms
    Λ2[α, :, i] .-= sum(Λ2[α, :, i]) / natoms
end

# Phonon modes at Γ
function compute_phonons(dynmat)
    M = reshape(DFTK.mass_matrix(Float64, model.atoms), 3*natoms, 3*natoms)

    phonon_res = eigen(reshape(dynmat, 3*natoms, 3*natoms), M)
    maximum(abs, imag(phonon_res.values)) > sqrt(eps(Float64)) &&
        @warn "Some eigenvalues of the dynamical matrix have a large imaginary part."

    signs = sign.(real(phonon_res.values))
    pulsations = signs .* sqrt.(abs.(real(phonon_res.values)))

    frequencies_invcm = pulsations ./ (2π * austrip(Unitful.c0 / 1u"cm"))
    frequencies_THz = pulsations ./ (2π * austrip(1u"THz"))

    (; pulsations, frequencies_invcm, frequencies_THz)
end

phonons_q0 = compute_phonons(K)

# note: currently not used
function na_correction(q, Zstar, ε∞)
    corr = zeros(3, natoms, 3, natoms)
    for s in 1:natoms, t in 1:natoms
        for α in 1:3, β in 1:3
            corr[α, s, β, t] = dot(q, Zstar[:, α, s]) * dot(q, Zstar[:, β, t])
        end
    end
    corr * 4π / model.unit_cell_volume / (q' * ε∞ * q)
end

# Print elementary response tensors
function new_section(name)
    println()
    @printf "==== %s ====\n" name
end
function print_dielectric(ϵ)
    for i in 1:3
        @printf "%8.2f %8.2f %8.2f\n" ϵ[i, 1] ϵ[i, 2] ϵ[i, 3]
    end
end
function print_elastic(C)
    C = C ./ austrip(1u"GPa")
    for i in 1:6
        @printf "%8.0f %8.0f %8.0f %8.0f %8.0f %8.0f\n" C[i, 1] C[i, 2] C[i, 3] C[i, 4] C[i, 5] C[i, 6]
    end
end
function print_piezoelectric(e)
    e = e ./ austrip(1u"C/m^2")
    println("            1          2          3          4          5          6")
    @printf "Ex %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n" e[1, 1] e[1, 2] e[1, 3] e[1, 4] e[1, 5] e[1, 6]
    @printf "Ey %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n" e[2, 1] e[2, 2] e[2, 3] e[2, 4] e[2, 5] e[2, 6]
    @printf "Ez %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n" e[3, 1] e[3, 2] e[3, 3] e[3, 4] e[3, 5] e[3, 6]
end

if occursin("ZnO", SYSTEM_NAME)
    new_section("Structural properties")
    println("Prop     Val")
    println("---- -------")
    @printf "a    %.3f Å\n" norm(model.lattice[:,1]) / austrip(1u"Å")
    @printf "c    %.3f Å\n" norm(model.lattice[:,3]) / austrip(1u"Å")
    u = model.positions[3][3] - model.positions[1][3]
    @printf "u    %.3f\n" u
elseif occursin("BaTiO3", SYSTEM_NAME)
    new_section("Structural properties")
    println("Prop     Val")
    println("---- -------")
    @printf "a    %.3f Å\n" norm(model.lattice[:,1]) / austrip(1u"Å")
    
    function vangle(v1, v2)
        acosd(dot(v1, v2) / (norm(v1) * norm(v2)))
    end

    @printf "α    %.2f°\n" vangle(model.lattice[:, 1], model.lattice[:, 2])

    Ti_displacement = model.positions[2] - model.positions[1] - [0.5, 0.5, -0.5]
    O_displacement = model.positions[3] - model.positions[1] - [0.5, 0.5, 0.0]

    println("Absolute displacements:")
    @printf "TiΔz %.3f Å\n" (model.lattice * Ti_displacement)[3] / austrip(1u"Å")
    @printf "O Δy %.3f Å\n" (model.lattice * O_displacement)[2] / austrip(1u"Å")
    @printf "O Δz %.3f Å\n" (model.lattice * O_displacement)[3] / austrip(1u"Å")

    println("Relative displacements:")
    @assert all(Ti_displacement .≈ Ti_displacement[1])
    @printf "TiΔx %.3f\n" Ti_displacement[1]
    @printf "O Δx %.3f\n" O_displacement[1]
    @printf "O Δz %.3f\n" O_displacement[3]
end

new_section("Phonon modes (LO only)")
@printf "%-4s %-11s %-10s\n" "Mode" "Freq [cm⁻¹]" "Freq [THz]"
println("---- ----------- ----------")
for (i, (freq_invcm, freq_THz)) in enumerate(zip(phonons_q0.frequencies_invcm, phonons_q0.frequencies_THz))
    @printf "%4d %11.1f %10.2f\n" i freq_invcm freq_THz
end

new_section("Clamped-ion dielectric tensor")
print_dielectric(ϵ_clamped_cart)

new_section("Clamped-ion elastic tensor [GPa]")
print_elastic(C_clamped)

new_section("Born effective charges [e]")
Zstar_mismatch = norm(Zstar1_cart - Zstar2_cart, Inf)
Zstar_avg = (Zstar1_cart + Zstar2_cart)/2
@printf "Mismatch between 2 routes: %.2e (absolute) %.2e (relative)\n" Zstar_mismatch Zstar_mismatch / norm(Zstar_avg, Inf)
println("Printing average")
for iatom in 1:natoms
    @printf "Atom %d          x          y          z\n" iatom
    @printf "    Ex %10.3f %10.3f %10.3f\n" Zstar_avg[1, 1, iatom] Zstar_avg[1, 2, iatom] Zstar_avg[1, 3, iatom]
    @printf "    Ey %10.3f %10.3f %10.3f\n" Zstar_avg[2, 1, iatom] Zstar_avg[2, 2, iatom] Zstar_avg[2, 3, iatom]
    @printf "    Ez %10.3f %10.3f %10.3f\n" Zstar_avg[3, 1, iatom] Zstar_avg[3, 2, iatom] Zstar_avg[3, 3, iatom]
end

new_section("Force-response internal strain [Ha/bohr]")
Λ_mismatch = norm(Λ1 - Λ2, Inf)
Λ_avg = (Λ1 + Λ2)/2
@printf "Mismatch between 2 routes: %.2e (absolute) %.2e (relative)\n" Λ_mismatch Λ_mismatch / norm(Λ_avg, Inf)
println("Printing average")
for iatom in 1:natoms
    @printf "Atom %d          1          2          3          4          5          6\n" iatom
    @printf "     x %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n" Λ_avg[1, iatom, 1] Λ_avg[1, iatom, 2] Λ_avg[1, iatom, 3] Λ_avg[1, iatom, 4] Λ_avg[1, iatom, 5] Λ_avg[1, iatom, 6]
    @printf "     y %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n" Λ_avg[2, iatom, 1] Λ_avg[2, iatom, 2] Λ_avg[2, iatom, 3] Λ_avg[2, iatom, 4] Λ_avg[2, iatom, 5] Λ_avg[2, iatom, 6]
    @printf "     z %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n" Λ_avg[3, iatom, 1] Λ_avg[3, iatom, 2] Λ_avg[3, iatom, 3] Λ_avg[3, iatom, 4] Λ_avg[3, iatom, 5] Λ_avg[3, iatom, 6]
end

new_section("Clamped-ion piezoelectric tensor [C/m²]")
e_mismatch = norm(e1_clamped_cart - e2_clamped_cart, Inf)
e_clamped_avg = (e1_clamped_cart + e2_clamped_cart)/2
@printf "Mismatch between 2 routes: %.2e (absolute) %.2e (relative)\n" (e_mismatch ./ austrip(1u"C/m^2")) e_mismatch / norm(e_clamped_avg, Inf)
println("Printing average")
print_piezoelectric(e_clamped_avg)

# Calculate relaxed-ion tensors
symmetrize(A) = Symmetric((A + A')/2)

K_22 = reshape(K, 3natoms, 3natoms)
K_22 = symmetrize(K_22)
K_22_eigvals = eigvals(K_22)
@assert sum(abs.(K_22_eigvals) .<= 1e-6) == 3 # ASR should give 3 zero modes
K_22_pinv = pinv(K_22; atol=1e-6)
Zstar_22_cart = reshape(Zstar_avg, 3, 3natoms)
Λ_22 = reshape(Λ_avg, 3natoms, 6)

χ_relaxed_cart = χ_clamped_cart + 1/model.unit_cell_volume * Zstar_22_cart * K_22_pinv * Zstar_22_cart'
ϵ_relaxed_cart = I + 4π * χ_relaxed_cart

C_relaxed = C_clamped - 1/model.unit_cell_volume * Λ_22' * K_22_pinv * Λ_22

e_relaxed = e_clamped_avg + 1/model.unit_cell_volume * Zstar_22_cart * K_22_pinv * Λ_22

# Print relaxed-ion tensors
new_section("Relaxed-ion dielectric tensor")
print_dielectric(ϵ_relaxed_cart)

new_section("Relaxed-ion elastic tensor [GPa]")
print_elastic(C_relaxed)

new_section("Relaxed-ion piezoelectric tensor [C/m²]")
print_piezoelectric(e_relaxed)