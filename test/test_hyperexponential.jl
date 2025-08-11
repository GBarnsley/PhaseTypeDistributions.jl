using PhaseTypeDistributions

@testset "Hyperexponential Distribution" begin
    
    @testset "Hyperexponential Constructor" begin
        # Test valid constructor
        λ = [1.0, 2.0, 3.0]
        α = [0.3, 0.4, 0.3]
        
        @test_nowarn Hyperexponential(λ, α)
        hyper = Hyperexponential(λ, α)
        
        # Test that it returns a MixtureModel
        @test hyper isa MixtureModel
        @test length(hyper.components) == 3
        
        # Test that components are Exponential distributions with correct rates
        for i in 1:3
            @test hyper.components[i] isa Exponential
            @test hyper.components[i].θ ≈ 1/λ[i]  # Exponential uses scale parameter θ = 1/rate
        end
        
        # Test that priors (mixing weights) are correct
        @test hyper.prior.p ≈ α
        
        # Test with compatible types (both float)
        λ_float = [1.0, 2.0, 3.0]
        α_float = [0.3, 0.4, 0.3]
        @test_nowarn Hyperexponential(λ_float, α_float)
        
        # Test two-component case
        λ_two = [1.5, 4.0]
        α_two = [0.6, 0.4]
        hyper_two = Hyperexponential(λ_two, α_two)
        @test length(hyper_two.components) == 2
        
        # Test single component (degenerate case)
        λ_single = [2.0]
        α_single = [1.0]
        hyper_single = Hyperexponential(λ_single, α_single)
        @test length(hyper_single.components) == 1
        @test hyper_single.components[1].θ ≈ 0.5  # 1/2.0
    end
    
    @testset "Hyperexponential Constructor Validation" begin
        # Test invalid probability vector (doesn't sum to 1)
        λ = [1.0, 2.0]
        α_invalid = [0.3, 0.4]  # Sums to 0.7, not 1.0
        @test_throws DomainError Hyperexponential(λ, α_invalid)
        
        # Test negative probabilities
        α_negative = [-0.1, 1.1]
        @test_throws DomainError Hyperexponential(λ, α_negative)
        
        # Test negative transition rates
        λ_negative = [-1.0, 2.0]
        α = [0.5, 0.5]
        @test_throws DomainError Hyperexponential(λ_negative, α)
        
        # Test zero transition rates
        λ_zero = [0.0, 2.0]
        @test_throws DomainError Hyperexponential(λ_zero, α)
        
        # Test dimension mismatch
        λ = [1.0, 2.0, 3.0]
        α_wrong_size = [0.5, 0.5]  # Should be length 3
        @test_throws DomainError Hyperexponential(λ, α_wrong_size)
        
        # Test empty inputs
        @test_throws DomainError Hyperexponential(Float64[], Float64[])
        
        # Test probability vector that sums to 0
        α_zero = [0.0, 0.0]
        λ = [1.0, 2.0]
        @test_throws DomainError Hyperexponential(λ, α_zero)
    end
    
end
