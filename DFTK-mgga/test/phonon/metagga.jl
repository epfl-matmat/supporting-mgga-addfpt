@testmodule PhononMetaGGA begin
using DFTK

function model_tested(lattice::AbstractMatrix, atoms::Vector{<:DFTK.Element},
                      positions::Vector{<:AbstractVector}; kwargs...)
    model_DFT(lattice, atoms, positions; functionals=r2SCAN(), kwargs...)
end
end

@testitem "Phonon: mGGA xc functional: comparison to supercell" #=
    =#    tags=[:phonon, :dont_test_mpi, :slow] #=
    =#    setup=[Phonon, PhononMetaGGA, TestCases] begin
    using .Phonon: test_frequencies
    using .PhononMetaGGA: model_tested
    test_frequencies(model_tested, TestCases.silicon)
end
