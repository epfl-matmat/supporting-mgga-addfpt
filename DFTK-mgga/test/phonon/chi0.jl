@testitem "Phonon: GGA chi0 at q != 0" tags=[:phonon] setup=[TestCases] begin
    using .TestCases
    using DFTK
    using LinearAlgebra

    testcase = TestCases.aluminium_primitive
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      symmetries=false, testcase.temperature, functionals=PBE())
    basis = PlaneWaveBasis(model, Ecut=10, kgrid=[1, 1, 2])
    scfres = self_consistent_field(basis)

    q = [0, 0, 0.5]
    qcart = DFTK.recip_vector_red_to_cart(model, q)
    δV = randn(complex(eltype(scfres.ρ)), size(scfres.ρ)...)
    # smoothen!
    δV_fourier = fft(basis, δV)
    δV = ifft(basis, δV_fourier ./ (norm.(DFTK.G_vectors_cart(basis)) .+ 1e-2).^2)

    δρ = apply_χ0(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues, δV; q, scfres.occupation_threshold).δρ

    # Build the supercell
    scfres_supercell = DFTK.cell_to_supercell(scfres)
    # TODO: crappy transfer of density, there should be a function in supercell.jl for this
    function transfer(arr)
        r_to_ir = Dict{Vec3{Float64},Int}()
        for ir in eachindex(DFTK.r_vectors(basis))
            r = DFTK.r_vectors(basis)[ir]
            r_to_ir[round.(r; digits=5)] = ir
        end
        arr_transferred = similar(arr, scfres_supercell.basis.fft_size..., model.n_spin_components)
        for ir in eachindex(DFTK.r_vectors(scfres_supercell.basis))
            r = DFTK.r_vectors(scfres_supercell.basis)[ir]
            r_uc = r .* [1, 1, 2]
            r_uc = r_uc[3] >= 1.0 ? r_uc .- [0, 0, 1] : r_uc
            i_uc = r_to_ir[round.(r_uc; digits=5)]
            arr_transferred[ir] = arr[i_uc]
        end
        arr_transferred
    end

    eiqx = exp.(im .* dot.(Ref(qcart), DFTK.r_vectors_cart(scfres_supercell.basis)))
    δρ_supercell = transfer(δρ)
    δρ_supercell .*= eiqx
    # δρ_supercell = 2 * real(δρ_supercell)

    δV_supercell = transfer(δV)
    δV_supercell .*= eiqx
    # δV_supercell = 2 * real(δV_supercell)

    doit(δV) = apply_χ0(scfres_supercell.ham, scfres_supercell.ψ, scfres_supercell.occupation,
                        scfres_supercell.εF, scfres_supercell.eigenvalues, δV;
                        scfres_supercell.occupation_threshold).δρ
    δρ_ref = doit(real.(δV_supercell)) .+ im .* doit(imag.(δV_supercell))

    @test δρ_supercell ≈ δρ_ref rtol=1e-7
    @show norm(δρ_supercell - δρ_ref) / norm(δρ_supercell)
end

@testitem "Phonon: meta-GGA chi0 at q != 0" tags=[:phonon] setup=[TestCases] begin
    using .TestCases
    using DFTK
    using LinearAlgebra

    testcase = TestCases.aluminium_primitive
    model = model_DFT(testcase.lattice, testcase.atoms, testcase.positions;
                      symmetries=false, testcase.temperature, functionals=r2SCAN())
    basis = PlaneWaveBasis(model, Ecut=10, kgrid=[1, 1, 2])
    scfres = self_consistent_field(basis)

    q = [0, 0, 0.5]
    qcart = DFTK.recip_vector_red_to_cart(model, q)
    δVρ = randn(complex(eltype(scfres.ρ)), size(scfres.ρ)...)
    # smoothen!
    δVρ_fourier = fft(basis, δVρ)
    δVρ = ifft(basis, δVρ_fourier ./ (norm.(DFTK.G_vectors_cart(basis)) .+ 1e-2).^2)

    δVτ = randn(complex(eltype(scfres.τ)), size(scfres.τ)...)
    δVτ_fourier = fft(basis, δVτ)
    δVτ = ifft(basis, δVτ_fourier ./ (norm.(DFTK.G_vectors_cart(basis)) .+ 1e-2).^2)

    (; δρ, δτ) = apply_χ0(scfres.ham, scfres.ψ, scfres.occupation, scfres.εF, scfres.eigenvalues, δVρ; q, scfres.occupation_threshold, δVτ)

    # Build the supercell
    scfres_supercell = DFTK.cell_to_supercell(scfres)
    # TODO: crappy transfer of density, there should be a function in supercell.jl for this
    function transfer(arr)
        r_to_ir = Dict{Vec3{Float64},Int}()
        for ir in eachindex(DFTK.r_vectors(basis))
            r = DFTK.r_vectors(basis)[ir]
            r_to_ir[round.(r; digits=5)] = ir
        end
        arr_transferred = similar(arr, scfres_supercell.basis.fft_size..., model.n_spin_components)
        for ir in eachindex(DFTK.r_vectors(scfres_supercell.basis))
            r = DFTK.r_vectors(scfres_supercell.basis)[ir]
            r_uc = r .* [1, 1, 2]
            r_uc = r_uc[3] >= 1.0 ? r_uc .- [0, 0, 1] : r_uc
            i_uc = r_to_ir[round.(r_uc; digits=5)]
            arr_transferred[ir] = arr[i_uc]
        end
        arr_transferred
    end

    eiqx = exp.(im .* dot.(Ref(qcart), DFTK.r_vectors_cart(scfres_supercell.basis)))
    δρ_supercell = transfer(δρ)
    δρ_supercell .*= eiqx
    δτ_supercell = transfer(δτ)
    δτ_supercell .*= eiqx

    δVρ_supercell = transfer(δVρ)
    δVρ_supercell .*= eiqx
    δVτ_supercell = transfer(δVτ)
    δVτ_supercell .*= eiqx

    doit(δVρ, δVτ) = apply_χ0(scfres_supercell.ham, scfres_supercell.ψ, scfres_supercell.occupation,
                        scfres_supercell.εF, scfres_supercell.eigenvalues, δVρ;
                        scfres_supercell.occupation_threshold, δVτ)
    ref_re = doit(real.(δVρ_supercell), real.(δVτ_supercell))
    ref_im = doit(imag.(δVρ_supercell), imag.(δVτ_supercell))
    δρ_ref = ref_re.δρ .+ im .* ref_im.δρ
    δτ_ref = ref_re.δτ .+ im .* ref_im.δτ

    @test δρ_supercell ≈ δρ_ref rtol=5e-7
    @show norm(δρ_supercell - δρ_ref) / norm(δρ_supercell)
    @test δτ_supercell ≈ δτ_ref rtol=5e-7
    @show norm(δτ_supercell - δτ_ref) / norm(δτ_supercell)
end
