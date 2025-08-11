abstract type FixedInitialPhaseTypeDistribution{T} <: PhaseTypeDistribution{T} end

struct Coxian{T<:Real, Tm <: AbstractMatrix{T}, Tv <: AbstractVector{T}} <: FixedInitialPhaseTypeDistribution{T}
    λ::Tv
    p::Tv
    #derived
    S::Tm
    α::Tv
    S⁰::Tv
    function Coxian{T}(λ::AbstractVector{T}, p::AbstractVector{T}; check_args::Bool=true) where T
        @check_args(
            Coxian,
            (p, all(x -> x ≥ zero(T) && x ≤ one(T), p), "p must be a vector of probabilities."),
            (λ, all(λ .> zero(T)), "λ must be a valid transition vector."),
            ((λ, p), length(λ) == (length(p) - 1), "p should have one less entry than λ.")
        )
        α = Vector{T}(undef, length(λ))
        α[1] = one(T)
        S = zeros(T, length(λ), length(λ))
        for i in eachindex(λ)
            S[i, i] = -λ[i]
            if i < length(λ)
                S[i, i + 1] = λ[i] * p[i]
            end
        end
        S⁰ = vec(-sum(S, dims = 2))
        new{T, typeof(S), typeof(α)}(λ, p, S, α, S⁰)
    end
end

@doc raw"""
    Coxian(λ, p; check_args=true)

Construct a Coxian phase-type distribution with transition rates `λ` and probabilities `p`.

A Coxian phase-type distribution represents the time until absorption in a finite-state continuous-time
Markov chain with one absorbing state. It is characterized by:
- A transition vector `λ` representing transition rates out of each transient state
- A vector of probabilities `p` where `1 - p` determines the probability of transitioning straight the absorbing state

These are used to construct the transition matrix `S` of a phase-type distribution
with these characteristics, where the initial probability vector `α` is set to `[1, 0, ..., 0]`.

This is essentially a wrapper around the `PhaseType` constructor allowing you to avoid manually constructing the transition matrix.

# Arguments
- `λ::AbstractMatrix`: The transition rates vector of length n representing the rates of leaving each transient state.
- `p::AbstractVector`: The probabilities vector of length n-1 representing probability (if transitioning) of moving to the next
state rather than directly to the absorbing state.
- `check_args::Bool=true`: Whether to validate input arguments.

# Mathematical Definition
See phase-type distribution.

# Examples
```julia
# Simple 2-phase Coxian distribution
λ = [2.0, 3.0]
p = [0.5, 0.5]
d = Coxian(λ, p)
# More complex Coxian distribution
λ = [1.0, 2.0, 3.0]
p = [0.2, 0.3, 0.5]
d = Coxian(λ, p)
# Construct a Coxian directly using a phase-type distribution
S = [
    -λ[1] λ[1] * p[1] 0.0;
    0.0 -λ[2] λ[2] * p[2];
    0.0 0.0 -λ[3]
]
α = [1.0, 0.0, 0.0]
d = PhaseType(S, α)
```

# Notes
- Exponential, Erlang, and hypoexponential distributions are special cases of the Coxian distribution
- The support is [0, ∞)
- Integer inputs are automatically converted to floating-point
"""
function Coxian(λ::AbstractVector{T}, p::AbstractVector{T}; check_args::Bool=true) where T<:Real
    Coxian{T}(λ, p; check_args=check_args)
end
function Coxian(λ::AbstractVector{Integer}, p::AbstractVector{Integer}; check_args::Bool=true)
    Coxian{eltype(float.(λ))}(float.(λ), float.(p); check_args=check_args)
end

struct FixedInitialPhaseTypeSampler{T <: Real} <: Sampleable{Univariate,Continuous}
    states::Vector{transition_state{T}}
end

function sampler(d::FixedInitialPhaseTypeDistribution{T}) where T
    (; S, S⁰) = d

    all_states = setup_states(S, S⁰, T)

    return FixedInitialPhaseTypeSampler(all_states)
end

function rand(rng::AbstractRNG, s::FixedInitialPhaseTypeSampler{T}) where T
    sample_states(rng, s.states, 1, T)
end

function rand(rng::AbstractRNG, d::FixedInitialPhaseTypeDistribution) 
    #really not that slow
    rand(rng, sampler(d))
end

function logpdf(d::FixedInitialPhaseTypeDistribution{T}, x::Real) where T
    insupport(d, x) ? log((exp(d.S * x)[1:1, :] * d.S⁰)[1, 1]) : -T(Inf)
end

function pdf(d::FixedInitialPhaseTypeDistribution{T}, x::Real) where T
    insupport(d, x) ? (exp(d.S * x)[1:1, :] * d.S⁰)[1, 1] : zero(T)
end

function cdf(d::FixedInitialPhaseTypeDistribution{T}, x::Real) where T
    insupport(d, x) ? 1 - sum(exp(d.S * x)[1:1, :]) : zero(T)
end
