@doc raw"""
    Hyperexponential(λ; check_args=true)

Construct a Hyperexponential distribution with transition rates `λ` and entry probability `α`.

A Hyperexponential distribution represents the time until absorption in a finite-state continuous-time
Markov chain with one absorbing state. It is characterized by:
- A transition vector `λ` representing transition rates out of each state and into the absorbing state
- An entry probability vector `α` representing the initial probabilities of being in each state

These are used to construct a Distributions.jl MixtureModel with Exponential components.

# Arguments
- `λ::AbstractMatrix`: The transition rates vector of length n representing the rates of moving to the final state.
- `α::AbstractVector`: The initial probability vector of length n representing the probabilities of starting in each state.
- `check_args::Bool=true`: Whether to validate input arguments.

# Mathematical Definition
See phase-type distribution.

# Examples
```julia
# Simple 2-phase Hyperexponential distribution
λ = [2.0, 3.0]
α = [0.5, 0.5]
d = Hyperexponential(λ, α)
# More complex Hyperexponential distribution
λ = [1.0, 2.0, 3.0]
α = [0.2, 0.5, 0.3]
d = Hyperexponential(λ, α)
# Construct a Hyperexponential directly using a phase-type distribution
S = [
    -λ[1] 0.0 0.0;
    0.0 -λ[2] 0.0;
    0.0 0.0 -λ[3]
]
d = PhaseType(S, α)
```

# Notes
- The support is [0, ∞)
"""
function Hyperexponential(λ::AbstractVector{T}, α::AbstractVector{T}; check_args::Bool=true) where T<:Real
    @check_args(
        Hyperexponential,
        (λ, length(λ) > 0, "λ must not be empty."),
        (α, all(x -> x ≥ zero(x), α) && sum(α) ≈ one(T), "α must be a probability vector."),
        (λ, all(λ .> zero(T)), "λ must be a valid transition vector."),
        ((α, λ), length(λ) == length(α), "λ and α must have the same length.")
    )
    MixtureModel(Exponential, 1 ./  λ, α)
end