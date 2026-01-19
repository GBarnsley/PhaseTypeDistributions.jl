using Test
using Distributions
using PhaseTypeDistributions
using Aqua

@testset "PhaseTypeDistributions.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(PhaseTypeDistributions)
    end

    # Include tests for specific distributions
    include("test_phasetype.jl")
    include("test_coxian.jl")
    include("test_hypoexponential.jl")
    include("test_hyperexponential.jl")
    include("test_comparisons.jl")

end
