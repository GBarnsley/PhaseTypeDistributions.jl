using Test
using Distributions
using LinearAlgebra

# Load the package from the parent directory
include("../src/PhaseTypeDistributions.jl")
using .PhaseTypeDistributions

@testset "PhaseTypeDistributions.jl" begin
    
    # Include tests for specific distributions
    include("test_phasetype.jl")
    
    # include("test_matrix_exponential.jl")
    # include("test_ph_mixture.jl")

end