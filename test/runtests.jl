using Test
using Distributions
using PhaseTypeDistributions

@testset "PhaseTypeDistributions.jl" begin
    
    # Include tests for specific distributions
    include("test_phasetype.jl")
    include("test_coxian.jl")
    include("test_hypoexponential.jl")
    include("test_hyperexponential.jl")
    include("test_comparisons.jl")
    
    # include("test_matrix_exponential.jl")
    # include("test_ph_mixture.jl")

end