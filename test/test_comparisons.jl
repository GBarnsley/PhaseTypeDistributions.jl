using Test
using Distributions
using PhaseTypeDistributions
using LinearAlgebra
using HypothesisTests
using Random

@testset "Comparisons with Distributions.jl" begin
    @testset "Erlang vs. PhaseType-Erlang (Hypoexponential)" begin
        # An Erlang(k, λ) distribution is equivalent to a Hypoexponential
        # distribution with k phases, where each phase has the same rate λ.

        k = 5
        λ = 2.0

        # Erlang from Distributions.jl - constructor is Erlang(shape, scale) where scale = 1/rate
        erlang_dist = Erlang(k, 1 / λ)

        # Equivalent PhaseType constructed via Hypoexponential
        ph_rates = fill(λ, k)
        ph_erlang = Hypoexponential(ph_rates)

        # Compare basic properties
        @test mean(erlang_dist) ≈ mean(ph_erlang)
        @test var(erlang_dist) ≈ var(ph_erlang)
        @test minimum(erlang_dist) == minimum(ph_erlang)
        @test maximum(erlang_dist) == maximum(ph_erlang)

        # Test pdf, logpdf, and cdf at various points
        test_points = [0.1, 0.5, 1.0, 2.5, 5.0]
        for x in test_points
            @test pdf(erlang_dist, x) ≈ pdf(ph_erlang, x)
            @test logpdf(erlang_dist, x) ≈ logpdf(ph_erlang, x)
            @test cdf(erlang_dist, x) ≈ cdf(ph_erlang, x)
        end
    end

    @testset "Hyperexponential (MixtureModel) vs. PhaseType-Hyperexponential" begin
        # A Hyperexponential distribution is a mixture of Exponential distributions,
        # which is equivalent to a PhaseType distribution with a diagonal S matrix.

        λ = [1.0, 5.0, 10.0]
        α = [0.2, 0.5, 0.3]

        # Hyperexponential from this package (which creates a MixtureModel)
        hyper_mix = Hyperexponential(λ, α)

        # Equivalent PhaseType constructed manually
        S_manual = diagm(-λ)
        pt_hyper = PhaseType(S_manual, α)

        # Compare basic properties
        @test mean(hyper_mix) ≈ mean(pt_hyper)
        @test var(hyper_mix) ≈ var(pt_hyper)

        # Test pdf, logpdf, and cdf at various points
        test_points = [0.05, 0.1, 0.5, 1.0, 2.0]
        for x in test_points
            @test pdf(hyper_mix, x) ≈ pdf(pt_hyper, x)
            @test logpdf(hyper_mix, x) ≈ logpdf(pt_hyper, x)
            @test cdf(hyper_mix, x) ≈ cdf(pt_hyper, x)
        end
    end

    @testset "Sampling Comparison" begin
        rng = MersenneTwister(1234)
        num_samples = 10000

        @testset "Erlang Sampling" begin
            k = 5
            λ = 2.0
            erlang_dist = Erlang(k, 1 / λ)
            ph_erlang = Hypoexponential(fill(λ, k))

            samples = rand(rng, ph_erlang, num_samples)

            ks_test = ApproximateOneSampleKSTest(samples, erlang_dist)
            @test pvalue(ks_test) > 0.15
        end

        @testset "Hyperexponential Sampling" begin
            λ = [1.0, 5.0, 10.0]
            α = [0.2, 0.5, 0.3]
            hyper_mix = Hyperexponential(λ, α)
            pt_hyper = PhaseType(diagm(-λ), α)

            samples = rand(rng, pt_hyper, num_samples)

            ks_test = ApproximateOneSampleKSTest(samples, hyper_mix)
            @test pvalue(ks_test) > 0.15
        end
    end
end
