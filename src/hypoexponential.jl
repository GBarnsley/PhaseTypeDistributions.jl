struct Hypoexponential{T <: Real, Tm <: AbstractMatrix{T}, Tv <: AbstractVector{T}} <:
       FixedInitialPhaseTypeDistribution{T}
    λ::Tv
    #derived
    S::Tm
    α::Tv
    S⁰::Tv
    function Hypoexponential{T}(λ::AbstractVector{T}; check_args::Bool = true) where {T}
        @check_args(Hypoexponential,
            (λ, length(λ) > 0, "λ must not be empty."),
            (λ, all(λ .> zero(T)), "λ must be a valid transition vector."))
        α = zeros(T, length(λ))
        α[1] = one(T)
        S = zeros(T, length(λ), length(λ))
        for i in eachindex(λ)
            S[i, i] = -λ[i]
            if i < length(λ)
                S[i, i + 1] = λ[i]
            end
        end
        S⁰ = zeros(T, length(λ))
        S⁰[end] = λ[end]
        new{T, typeof(S), typeof(α)}(λ, S, α, S⁰)
    end
end

@doc raw"""
    Hypoexponential(λ; check_args=true)

Construct a Hypoexponential distribution with transition rates `λ`.

A Hypoexponential distribution represents the time until absorption in a finite-state continuous-time
Markov chain with one absorbing state. It is characterized by:
- A transition vector `λ` representing transition rates out of each transient state and into the next state

These are used to construct the transition matrix `S` of a phase-type distribution
with these characteristics, where the initial probability vector `α` is set to `[1, 0, ..., 0]`.

This is essentially a wrapper around the `PhaseType` constructor allowing you to avoid manually constructing the transition matrix.

# Arguments
- `λ::AbstractMatrix`: The transition rates vector of length n representing the rates of leaving each transient state.
- `check_args::Bool=true`: Whether to validate input arguments.

# Mathematical Definition
See phase-type distribution.

# Examples
```julia
# Simple 2-phase Hypoexponential distribution
λ = [2.0, 3.0]
d = Hypoexponential(λ)
# More complex Hypoexponential distribution
λ = [1.0, 2.0, 3.0]
d = Hypoexponential(λ)
# Construct a Hypoexponential directly using a phase-type distribution
S = [
    -λ[1] λ[1] 0.0;
    0.0 -λ[2] λ[2];
    0.0 0.0 -λ[3]
]
α = [1.0, 0.0, 0.0]
d = PhaseType(S, α)
```

# Notes
- Exponential, and Erlang distributions are special cases of the Hypoexponential distribution
- The support is [0, ∞)
- Integer inputs are automatically converted to floating-point
"""
function Hypoexponential(λ::AbstractVector{T}; check_args::Bool = true) where {T <: Real}
    Hypoexponential{T}(λ; check_args = check_args)
end
function Hypoexponential(λ::AbstractVector{Integer}; check_args::Bool = true)
    Hypoexponential{eltype(float.(λ))}(float.(λ); check_args = check_args)
end

struct HypoexponentialSampler{T <: Real} <: Sampleable{Univariate, Continuous}
    dists::Vector{Exponential{T}}
end

function sampler(d::Hypoexponential)
    #Should make this the sampleables of the Exponential instead of the exponentials
    all_dists = Exponential.(1 ./ d.λ)

    return HypoexponentialSampler(all_dists)
end

function rand(rng::AbstractRNG, s::HypoexponentialSampler)
    sum(rand.(rng, s.dists))
end

function rand(rng::AbstractRNG, d::Hypoexponential)
    sum(rand.(rng, Exponential.(1 ./ d.λ)))
end

#other functions
mean(d::Hypoexponential) = sum(1 ./ d.λ)
var(d::Hypoexponential) = sum((1 ./ d.λ) .^ 2)
skewness(d::Hypoexponential) = 2.0 * sum(1 ./ (d.λ .^ 3)) / (sum(1 ./ (d.λ .^ 2))^(3 / 2))
