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
        S⁰ = -S[end, :]
        new{T, typeof(S), typeof(α)}(S, α, S⁰)
    end
end

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
    (; S, α) = d

    shape = size(S, 1)
    starting_states = findall(α .> zero(T))
    α_dist = DiscreteNonParametric(starting_states, α[starting_states])

    all_states = Vector{transition_state{T}}(undef, shape)

    for i in axes(S, 1)
        next = findall(S[i, :] .> zero(T))
        dists = Exponential.(S[i, next])

        direct_absorbing = -sum(S[i, :])
        if direct_absorbing > zero(T)
            push!(dists, Exponential(direct_absorbing))
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

function logpdf(d::PhaseType, x::Real)
    (; S, α, S⁰) = d
    
    log((α' * exp(S * x) * S⁰)[1, 1])
end

function pdf(d::PhaseType, x::Real)
    (; S, α, S⁰) = d

    (α' * exp(S * x) * S⁰)[1, 1]
end

function cdf(d::PhaseType, x::Real)
    (; S, α) = d

    1 - sum(α' * exp(S * x))
end

function quantile(d::PhaseType, q::Real)
    error("Quantile function not implemented for PhaseType distributions. import Optimization.jl for quantile approximation.")
end

#other functions

minimum(d::PhaseType{T}) = Zero(T)
maximum(d::PhaseType) = Inf

mean(d::PhaseType{T}) = -d.α' * inv(d.S) * ones(T, size(d.S, 1))
var(d::PhaseType{T}) = 2 * d.α' * inv(d.S) * inv(d.S) * ones(T, size(d.S, 1)) - (d.α' * inv(d.S) * ones(T, size(d.S, 1)))^2
mgf(d::PhaseType, t) = -d.α' * inv(d.S + t * I) * d.S⁰
cf(d::PhaseType, t) = -d.α' * inv(d.S + t * im * I) * d.S⁰
