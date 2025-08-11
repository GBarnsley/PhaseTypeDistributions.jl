using PhaseTypeDistributions

@testset "Hypoexponential Distribution" begin
    
    @testset "Hypoexponential Constructor" begin
        # Test valid constructor
        λ = [1.0, 2.0, 3.0]
        
        @test_nowarn Hypoexponential(λ)
        hypo = Hypoexponential(λ)
        @test hypo.λ == λ
        
        # Test that it creates proper S matrix structure (tandem/series)
        @test size(hypo.S) == (3, 3)
        @test hypo.S[1, 1] == -λ[1]
        @test hypo.S[2, 2] == -λ[2]
        @test hypo.S[3, 3] == -λ[3]
        @test hypo.S[1, 2] == λ[1]  # Flow from state 1 to 2
        @test hypo.S[2, 3] == λ[2]  # Flow from state 2 to 3
        @test hypo.S[1, 3] == 0.0   # No direct transition from state 1 to 3
        @test hypo.S[3, 1] == 0.0   # No backward transitions
        @test hypo.S[3, 2] == 0.0   # No backward transitions
        
        # Test initial probability vector (should be [1, 0, 0, ...])
        @test hypo.α[1] == 1.0
        @test all(hypo.α[2:end] .== 0.0)
        
        # Test exit vector S⁰ (only from last state)
        expected_s0 = [0.0, 0.0, λ[3]]
        @test hypo.S⁰ ≈ expected_s0
        
        # Test integer inputs work correctly 
        λ_int = [1, 2, 3]
        @test_nowarn Hypoexponential(λ_int)
        hypo_int = Hypoexponential(λ_int)
        # The implementation may or may not convert to float internally
        @test eltype(rand(hypo_int)) == Float64
        
        # Test single phase
        λ_single = [4.0]
        hypo_single = Hypoexponential(λ_single)
        @test hypo_single.S == reshape([-4.0], 1, 1)
        @test hypo_single.α == [1.0]
        @test hypo_single.S⁰ == [4.0]
    end
    
    @testset "Hypoexponential Constructor Validation" begin
        # Test negative transition rates
        λ_negative = [-1.0, 2.0, 3.0]
        @test_throws DomainError Hypoexponential(λ_negative)
        
        # Test zero transition rates
        λ_zero = [1.0, 0.0, 3.0]
        @test_throws DomainError Hypoexponential(λ_zero)
        
        # Test empty input
        @test_throws DomainError Hypoexponential(Float64[])
    end
    
    @testset "Hypoexponential Distribution Properties" begin
        λ = [2.0, 1.0, 3.0]
        hypo = Hypoexponential(λ)
        
        # Test that it's a proper distribution
        @test hypo isa PhaseTypeDistributions.FixedInitialPhaseTypeDistribution
        @test minimum(hypo) == 0.0
        @test maximum(hypo) == Inf
        
        # Test basic properties work
        @test_nowarn mean(hypo)
        @test_nowarn var(hypo)
        @test mean(hypo) > 0
        @test var(hypo) > 0
        
        # For Hypoexponential, mean should be sum of 1/λᵢ
        expected_mean = sum(1 ./ λ)
        @test mean(hypo) ≈ expected_mean
        
        # Test evaluation at specific points
        @test_nowarn pdf(hypo, 0.5)
        @test_nowarn cdf(hypo, 0.5)
        @test pdf(hypo, 0.0) ≥ 0
        @test pdf(hypo, 1.0) ≥ 0
        @test 0 ≤ cdf(hypo, 1.0) ≤ 1
        
        # Test support
        @test insupport(hypo, 0.0)
        @test insupport(hypo, 1.0)
        @test !insupport(hypo, -1.0)
    end
    
    @testset "Hypoexponential Special Cases" begin
        # Test single exponential (equivalent to Exponential(1/λ))
        λ_single = [2.5]
        hypo_single = Hypoexponential(λ_single)
        exp_dist = Exponential(1/2.5)
        
        # Should have same mean and variance
        @test mean(hypo_single) ≈ mean(exp_dist)
        @test var(hypo_single) ≈ var(exp_dist)
        
        # Test identical rates (Erlang distribution)
        λ_identical = [3.0, 3.0, 3.0]
        hypo_erlang = Hypoexponential(λ_identical)
        
        # Mean should be n/λ = 3/3 = 1
        @test mean(hypo_erlang) ≈ 1.0
        
        # Test two-phase case
        λ_two = [1.0, 2.0]
        hypo_two = Hypoexponential(λ_two)
        @test mean(hypo_two) ≈ 1.0 + 0.5  # 1/1 + 1/2
    end
    
    @testset "Hypoexponential Sampling" begin
        λ = [1.5, 2.5, 3.5]
        hypo = Hypoexponential(λ)
        
        # Test that sampling works
        @test_nowarn rand(hypo)
        @test_nowarn rand(hypo, 10)
        
        # Test samples are non-negative
        samples = rand(hypo, 100)
        @test all(samples .≥ 0)
        
        # Test with specific RNG for reproducibility
        using Random
        rng = MersenneTwister(456)
        sample1 = rand(rng, hypo)
        rng = MersenneTwister(456)
        sample2 = rand(rng, hypo)
        @test sample1 == sample2
        
        # For large samples, mean should be close to theoretical
        large_samples = rand(hypo, 10000)
        sample_mean = mean(large_samples)
        theoretical_mean = sum(1 ./ λ)
        @test abs(sample_mean - theoretical_mean) < 0.1  # Within reasonable tolerance
    end
    
    @testset "Hypoexponential vs PhaseType Equivalence" begin
        # Test that Hypoexponential is equivalent to manually constructed PhaseType
        λ = [1.0, 2.0, 3.0]
        hypo = Hypoexponential(λ)
        
        # Manually construct equivalent PhaseType
        S_manual = [
            -1.0  1.0  0.0;
             0.0 -2.0  2.0;
             0.0  0.0 -3.0
        ]
        α_manual = [1.0, 0.0, 0.0]
        pt_manual = PhaseType(S_manual, α_manual)
        
        # Should be equivalent
        @test hypo.S ≈ pt_manual.S
        @test hypo.α ≈ pt_manual.α
        @test hypo.S⁰ ≈ pt_manual.S⁰
        
        # Should have same distribution properties
        @test mean(hypo) ≈ mean(pt_manual)
        @test var(hypo) ≈ var(pt_manual)
    end
end
