@testset "Coxian Distribution" begin
    @testset "Coxian Constructor" begin
        # Test valid constructor
        λ = [2.0, 3.0, 1.0]
        p = [0.5, 0.7]

        @test_nowarn Coxian(λ, p)
        cox = Coxian(λ, p)
        @test cox.λ == λ
        @test cox.p == p

        # Test that it creates proper S matrix structure
        @test size(cox.S) == (3, 3)
        @test cox.S[1, 1] == -λ[1]
        @test cox.S[2, 2] == -λ[2]
        @test cox.S[3, 3] == -λ[3]
        @test cox.S[1, 2] == λ[1] * p[1]
        @test cox.S[2, 3] == λ[2] * p[2]
        @test cox.S[1, 3] == 0.0  # No direct transition from state 1 to 3

        # Test initial probability vector (should be [1, 0, 0, ...])
        @test cox.α[1] == 1.0
        @test all(cox.α[2:end] .== 0.0)

        # Test exit vector S⁰
        expected_s0 = [λ[1] * (1 - p[1]), λ[2] * (1 - p[2]), λ[3]]
        @test cox.S⁰ ≈ expected_s0

        # Test integer inputs are converted to float
        λ_int = [2, 3, 1]
        p_int = [1, 1]  # Edge case: all probability goes to next state
        @test_nowarn Coxian(λ_int, p_int)

        # Test single phase (just one λ, no p)
        λ_single = [5.0]
        p_empty = Float64[]
        @test_nowarn Coxian(λ_single, p_empty)
        cox_single = Coxian(λ_single, p_empty)
        @test cox_single.S == reshape([-5.0], 1, 1)
        @test cox_single.α == [1.0]
        @test cox_single.S⁰ == [5.0]
    end

    @testset "Coxian Constructor Validation" begin
        # Test invalid probability values (> 1)
        λ = [2.0, 3.0]
        p_invalid = [1.5]
        @test_throws DomainError Coxian(λ, p_invalid)

        # Test negative probabilities
        p_negative = [-0.1]
        @test_throws DomainError Coxian(λ, p_negative)

        # Test negative transition rates
        λ_negative = [-1.0, 2.0]
        p = [0.5]
        @test_throws DomainError Coxian(λ_negative, p)

        # Test zero transition rates
        λ_zero = [0.0, 2.0]
        @test_throws DomainError Coxian(λ_zero, p)

        # Test dimension mismatch (p should have one less entry than λ)
        λ = [2.0, 3.0, 1.0]
        p_wrong_size = [0.5]  # Should be length 2
        @test_throws DomainError Coxian(λ, p_wrong_size)

        p_too_long = [0.5, 0.7, 0.3]  # Should be length 2
        @test_throws DomainError Coxian(λ, p_too_long)

        # Test empty inputs
        @test_throws DomainError Coxian(Float64[], Float64[])
    end

    @testset "Coxian Distribution Properties" begin
        λ = [1.0, 2.0, 3.0]
        p = [0.6, 0.8]
        cox = Coxian(λ, p)

        # Test that it's a proper distribution
        @test cox isa PhaseTypeDistributions.FixedInitialPhaseTypeDistribution
        @test minimum(cox) == 0.0
        @test maximum(cox) == Inf

        # Test basic properties work (just that they don't error)
        @test_nowarn mean(cox)
        @test_nowarn var(cox)
        @test mean(cox) > 0
        @test var(cox) > 0

        # Test evaluation at specific points
        @test_nowarn pdf(cox, 0.5)
        @test_nowarn cdf(cox, 0.5)
        @test pdf(cox, 0.0) ≥ 0
        @test pdf(cox, 1.0) ≥ 0
        @test 0 ≤ cdf(cox, 1.0) ≤ 1

        # Test support
        @test insupport(cox, 0.0)
        @test insupport(cox, 1.0)
        @test !insupport(cox, -1.0)
    end

    @testset "Coxian Special Cases" begin
        # Test single exponential (equivalent to Exponential(1/λ))
        λ_single = [3.0]
        cox_single = Coxian(λ_single, Float64[])
        exp_dist = Exponential(1 / 3.0)

        # Should have same mean
        @test mean(cox_single) ≈ mean(exp_dist)
        @test var(cox_single) ≈ var(exp_dist)

        # Test with all probabilities = 1 (pure series)
        λ = [1.0, 2.0, 3.0]
        p = [1.0, 1.0]  # All transitions go to next state
        cox_series = Coxian(λ, p)

        # This should be equivalent to Hypoexponential
        hypo = Hypoexponential(λ)
        @test cox_series.S ≈ hypo.S
        @test cox_series.α ≈ hypo.α

        # Test with all probabilities = 0 (parallel structure)
        p_zero = [0.0, 0.0]
        cox_parallel = Coxian(λ, p_zero)
        @test cox_parallel.S[1, 2] == 0.0
        @test cox_parallel.S[2, 3] == 0.0
        @test cox_parallel.S⁰ ≈ λ  # All exit rates equal to λ
    end

    @testset "Coxian Sampling" begin
        λ = [2.0, 3.0]
        p = [0.7]
        cox = Coxian(λ, p)

        # Test that sampling works
        @test_nowarn rand(cox)
        @test_nowarn rand(cox, 10)

        # Test samples are non-negative
        samples = rand(cox, 100)
        @test all(samples .≥ 0)

        # Test with specific RNG for reproducibility
        using Random
        rng = MersenneTwister(123)
        sample1 = rand(rng, cox)
        rng = MersenneTwister(123)
        sample2 = rand(rng, cox)
        @test sample1 == sample2
    end
end
