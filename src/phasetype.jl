abstract type PhaseTypeDistribution <: ContinuousUnivariateDistribution end

struct PhaseType{T<:Real, Tm <: AbstractMatrix{T}, Tv <: AbstractVector{T}} <: PhaseTypeDistribution
    S::Tm
    α::Tv

    #derived
    S⁰::Tv
    function PhaseType{T}(S::AbstractMatrix{T}, α::AbstractVector{T}; check_args::Bool=true) where T
        @check_args(
            PhaseType,
            (α, all(x -> x ≥ zero(x), α) && sum(α) ≈ one(T), "α must be a probability vector."),
            (S, all(diag(S) .< zero(T)), "S must be a valid transition matrix."),
            (S, all(S[.!I(size(S, 1))] .≥ zero(T)), "S must be a valid transition matrix.")
            #(S, all(sum(S[1:(end - 1), :], dims=2) .≈ zero(T)), "S must be a valid transition matrix.") Not needed for this save for hypo
        )
        S⁰ = vec(-sum(S, dims = 2))
        new{T, typeof(S), typeof(α)}(S, α, S⁰)
    end
end

@doc raw"""
    PhaseType(S, α; check_args=true)

Construct a phase-type distribution with transition matrix `S` and initial probability vector `α`.

A phase-type distribution represents the time until absorption in a finite-state continuous-time 
Markov chain with one absorbing state. It is characterized by:
- A transition matrix `S` representing transitions between transient states
- An initial probability vector `α` over the transient states
- An absorption vector `S⁰` (computed automatically) representing absorption rates

# Arguments
- `S::AbstractMatrix`: The sub-generator matrix of size n×n representing transitions between 
  transient states. Must have negative diagonal elements and non-negative off-diagonal elements.
  Should not include the final (absorbing) state, this is inferred from the structure of `S`.
- `α::AbstractVector`: Initial probability vector of length n. Must be non-negative and sum to 1.
- `check_args::Bool=true`: Whether to validate input arguments.

# Mathematical Definition
The phase-type distribution has:
- **PDF**: ``f(x) = α^T e^{Sx} S^0`` for ``x ≥ 0``
- **CDF**: ``F(x) = 1 - α e^{Sx} \mathbf{1}`` for ``x ≥ 0``
- **Mean**: ``μ = -α S^{-1} \mathbf{1}``
- **Variance**: ``σ^2 = 2α S^{-2} \mathbf{1} - μ^2``

where ``S^0 = -S\mathbf{1}`` is the absorption rate vector.

# Examples
```julia
# Simple 2-phase Erlang distribution
S = [-2.0 2.0; 0.0 -2.0]
α = [1.0, 0.0]
d = PhaseType(S, α)

# Mixed exponential distribution
S = [-1.0 0.0; 0.0 -2.0]
α = [0.7, 0.3]
d = PhaseType(S, α)

# Hypoexponential distribution
S = [-1.0 1.0; 0.0 -2.0]
α = [1.0, 0.0]
d = PhaseType(S, α)
```

# Notes
- Phase-type distributions are dense in the class of positive-valued distributions
- They include exponential, Erlang, hyperexponential, and hypoexponential distributions as special cases
- The support is [0, ∞)
- Integer inputs are automatically converted to floating-point
"""
function PhaseType(S::AbstractMatrix{T}, α::AbstractVector{T}; check_args::Bool=true) where T<:Real
    PhaseType{T}(S, α; check_args=check_args)
end
function PhaseType(S::AbstractMatrix{Integer}, α::AbstractVector{Integer}; check_args::Bool=true)
    PhaseType{eltype(float.(α))}(float.(S), float.(α); check_args=check_args)
end

## Sampling
struct transition_state{T <: Real}
    dist::Vector{Exponential{T}}
    next::Vector{Int}
end

struct PhaseTypeSampler{T <: Real} <: Sampleable{Univariate,Continuous}
    α_dist::DiscreteNonParametric
    states::Vector{transition_state{T}}
end

function rand(rng::AbstractRNG, t::transition_state)
    x = rand.(rng, t.dist)
    time, i = findmin(x)
    return (time, t.next[i])
end

function sampler(d::PhaseType{T}) where T
    (; S, α, S⁰) = d

    shape = size(S, 1)
    starting_states = findall(α .> zero(T))
    α_dist = DiscreteNonParametric(starting_states, α[starting_states])

    all_states = Vector{transition_state{T}}(undef, shape)

    for i in axes(S, 1)
        next = findall(S[i, :] .> zero(T))
        dists = Exponential.(1 ./ S[i, next])  # Exponential(θ) has mean θ, so θ = 1/rate

        if S⁰[i] > zero(T)
            push!(dists, Exponential(1 / S⁰[i]))
            push!(next, 0)
        end
        all_states[i] = transition_state(dists, next)
    end
    return PhaseTypeSampler(α_dist, all_states)
end

function rand(rng::AbstractRNG, s::PhaseTypeSampler{T}) where T
    states = s.states
    current_state_index = rand(rng, s.α_dist)
    x = zero(T)
    while true
        x⁺, current_state_index = rand(rng, states[current_state_index])
        x += x⁺
        if current_state_index == 0
            return x
        end
    end
end

function rand(rng::AbstractRNG, d::PhaseType) 
    #really not that slow
    rand(rng, sampler(d))
end

function logpdf(d::PhaseType{T}, x::Real) where T
    insupport(d, x) ? log((d.α' * exp(d.S * x) * d.S⁰)[1, 1]) : -T(Inf)
end

function pdf(d::PhaseType{T}, x::Real) where T
    insupport(d, x) ? (d.α' * exp(d.S * x) * d.S⁰)[1, 1] : zero(T)
end

function cdf(d::PhaseType{T}, x::Real) where T
    insupport(d, x) ? 1 - sum(d.α' * exp(d.S * x)) : zero(T)
end

function quantile(d::PhaseType, q::Real)
    error("Quantile function not implemented for PhaseType distributions. import Optimization.jl for quantile approximation.")
end

#other functions

insupport(d::PhaseType, x::Real) = x ≥ zero(x)

minimum(d::PhaseType{T}) where T = zero(T)
maximum(d::PhaseType{T}) where T = T(Inf)

mean(d::PhaseType{T}) where T = (-d.α' * inv(d.S) * ones(T, size(d.S, 1)))[1]
var(d::PhaseType{T}) where T = (2 * d.α' * inv(d.S) * inv(d.S) * ones(T, size(d.S, 1)) - (d.α' * inv(d.S) * ones(T, size(d.S, 1)))^2)[1]
mgf(d::PhaseType, t) = -(d.α' * inv(d.S + t * I) * d.S⁰)[1]
cf(d::PhaseType, t) = -(d.α' * inv(d.S + t * im * I) * d.S⁰)[1]
