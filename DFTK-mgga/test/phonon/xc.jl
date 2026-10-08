@testitem "Phonon: LDA: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    # Values computed offline with automatic differentiation.
    ω_ref = [ -0.002394568935772381
              -0.0009483244516830787
              -0.0009483244516742963
              -0.0007011496681061224
              -2.3342510049395543e-6
               1.2772524101950583e-6
               9.236368313346967e-6
               0.0004853028802285156
               0.00048530288023371785
               0.0005162064130028024
               0.0005170848409142678
               0.0006579053502793632
               0.0008427616671355543
               0.0008427616671396212
               0.0012763347783044433
               0.0012763347783092767
               0.0015604654945598588
               0.0015609820766927637 ]

    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: LDA+NLCC: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    # Values computed offline with automatic differentiation.
    ω_ref = [ -0.002291246044167315
              -0.0008322141414252373
              -0.0008322141414235875
              -0.0006277141258580046
              -9.265838548786679e-9
               2.7728753822878845e-9
               1.723854614577606e-8
               0.0005132198029985638
               0.000513219803002495
               0.0005311935317559987
               0.0005311935317559987
               0.0006681073906670919
               0.0008491725632174406
               0.0008491725632198118
               0.0012978110205401107
               0.001297811020540865
               0.0015922296328008808
               0.0015922296328024783  ]

    Al = ElementPsp(:Al, load_psp(TestCases.aluminium_primitive.psp_upf))
    aluminium_primitive = merge(TestCases.aluminium_primitive, (; atoms=[Al]))
    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, aluminium_primitive; ω_ref)
end

@testitem "Phonon: LDA+NLCC: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK
    Al = ElementPsp(:Al, load_psp(TestCases.aluminium_primitive.psp_upf))
    aluminium_primitive = merge(TestCases.aluminium_primitive, (; atoms=[Al]))
    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, aluminium_primitive)
end

@testitem "Phonon: LDA: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    model_LDA(args...; kwargs...) = model_DFT(args...; functionals=LDA(), kwargs...)
    test_frequencies(model_LDA, TestCases.aluminium_primitive)
end

@testitem "Phonon: PBE: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    model_PBE(args...; kwargs...) = model_DFT(args...; functionals=PBE(), kwargs...)
    test_frequencies(model_PBE, TestCases.aluminium_primitive)
end

@testitem "Phonon: meta-GGA: comparison to ref testcase" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    ω_ref = [-0.0024332543736926456, -0.001061845350861051, -0.0010618453508592818, -0.0007133544092323016, 9.140249667992474e-5, 0.00018250715223250055, 0.00022851939684138996, 0.0005102842378773082, 0.0005102842378803313, 0.000564614217541773, 0.0005646142175437434, 0.0006762690131256396, 0.0008685068045171067, 0.0008685068045189695, 0.0013112648228931307, 0.0013112648228946451, 0.0016012468491831422, 0.001601246849183899]

    model_r2SCAN(args...; kwargs...) = model_DFT(args...; functionals=r2SCAN(), kwargs...)
    test_frequencies(model_r2SCAN, TestCases.aluminium_primitive; ω_ref)
end

@testitem "Phonon: meta-GGA: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    model_r2SCAN(args...; kwargs...) = model_DFT(args...; functionals=r2SCAN(), kwargs...)
    test_frequencies(model_r2SCAN, TestCases.aluminium_primitive)
end

@testitem "Phonon: meta-GGA+NLCC: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] setup=[Phonon, TestCases] begin
    using .Phonon: test_frequencies
    using DFTK

    psp = load_psp(joinpath(@__DIR__, "..", "pseudos", "Al_m.upf"))
    @test DFTK.has_core_density(psp)
    @test DFTK.has_core_kinetic_energy_density(psp)
    Al = ElementPsp(:Al, psp)
    aluminium_primitive = merge(TestCases.aluminium_primitive, (; atoms=[Al]))
    model_r2SCAN(args...; kwargs...) = model_DFT(args...; functionals=r2SCAN(), kwargs...)
    test_frequencies(model_r2SCAN, aluminium_primitive)
end

@testitem "Phonon: GGA kernel at q != 0" tags=[:phonon] setup=[TestCases] begin
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
    δρ = randn(complex(eltype(scfres.ρ)), size(scfres.ρ)...)
    # smoothen!
    δρ_fourier = fft(basis, δρ)
    δρ = ifft(basis, δρ_fourier ./ (norm.(DFTK.G_vectors_cart(basis)) .+ 1e-2).^2)

    function apply_xc_kernel(basis, δρ; kwargs...)
        xc_term = only(filter(t -> t isa DFTK.TermXc, basis.terms))
        DFTK.apply_kernel(xc_term, basis, δρ; kwargs...)
    end
    δV = apply_xc_kernel(basis, δρ; scfres.ρ, q).δVρ

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
    ρ_supercell = transfer(scfres.ρ)
    @test scfres_supercell.ρ ≈ ρ_supercell

    eiqx = exp.(im .* dot.(Ref(qcart), DFTK.r_vectors_cart(scfres_supercell.basis)))
    δρ_supercell = transfer(δρ)
    δρ_supercell .*= eiqx
    # δρ_supercell = 2 * real(δρ_supercell)

    δV_supercell = transfer(δV)
    δV_supercell .*= eiqx
    # δV_supercell = 2 * real(δV_supercell)

    δV_ref = apply_xc_kernel(scfres_supercell.basis, δρ_supercell; ρ=ρ_supercell).δVρ

    @test δV_supercell ≈ δV_ref rtol=1e-12
    @show norm(δV_supercell - δV_ref) / norm(δV_supercell)
end

@testitem "Phonon: meta-GGA kernel at q != 0" tags=[:phonon] setup=[TestCases] begin
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
    δρ = randn(complex(eltype(scfres.ρ)), size(scfres.ρ)...)
    # smoothen!
    δρ_fourier = fft(basis, δρ)
    δρ = ifft(basis, δρ_fourier ./ (norm.(DFTK.G_vectors_cart(basis)) .+ 1e-2).^2)
    δτ = randn(complex(eltype(scfres.τ)), size(scfres.τ)...)
    # smoothen!
    δτ_fourier = fft(basis, δτ)
    δτ = ifft(basis, δτ_fourier ./ (norm.(DFTK.G_vectors_cart(basis)) .+ 1e-2).^2)

    function apply_xc_kernel(basis, δρ; kwargs...)
        xc_term = only(filter(t -> t isa DFTK.TermXc, basis.terms))
        DFTK.apply_kernel(xc_term, basis, δρ; kwargs...)
    end
    ker = apply_xc_kernel(basis, δρ; scfres.ρ, q, scfres.τ, δτ)
    δVρ = ker.δVρ
    δVτ = ker.δVτ

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
    ρ_supercell = transfer(scfres.ρ)
    @test scfres_supercell.ρ ≈ ρ_supercell

    eiqx = exp.(im .* dot.(Ref(qcart), DFTK.r_vectors_cart(scfres_supercell.basis)))
    δρ_supercell = transfer(δρ)
    δρ_supercell .*= eiqx
    # δρ_supercell = 2 * real(δρ_supercell)

    τ_supercell = transfer(scfres.τ)
    @test scfres_supercell.τ ≈ τ_supercell

    δτ_supercell = transfer(δτ)
    δτ_supercell .*= eiqx

    δVρ_supercell = transfer(δVρ)
    δVρ_supercell .*= eiqx
    # δV_supercell = 2 * real(δV_supercell)

    δVτ_supercell = transfer(δVτ)
    δVτ_supercell .*= eiqx

    ker_ref = apply_xc_kernel(scfres_supercell.basis, δρ_supercell; ρ=ρ_supercell, τ=τ_supercell, δτ=δτ_supercell)
    δVρ_ref = ker_ref.δVρ
    δVτ_ref = ker_ref.δVτ

    @test δVρ_supercell ≈ δVρ_ref rtol=1e-12
    @test δVτ_supercell ≈ δVτ_ref rtol=1e-12
    # @show norm(δV_supercell - δV_ref) / norm(δV_supercell)
end
