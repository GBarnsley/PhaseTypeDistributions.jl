using Test
using Distributions
using PhaseTypeDistributions
using Aqua, JET, Pkg

@testset "PhaseTypeDistributions.jl" begin

    @testset "Aqua" begin
        Aqua.test_all(PhaseTypeDistributions)
    end

    if pkgversion(JET) ≥ v"0.9"
        @testset "JET" begin
            JET.test_package(PhaseTypeDistributions)
        end
    end

    # Include tests for specific distributions
    include("test_phasetype.jl")
    include("test_coxian.jl")
    include("test_hypoexponential.jl")
    include("test_hyperexponential.jl")
    include("test_comparisons.jl")

end
