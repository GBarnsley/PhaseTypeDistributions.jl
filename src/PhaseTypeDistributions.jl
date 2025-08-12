module PhaseTypeDistributions

export PhaseType, Coxian, Hypoexponential, Hyperexponential

import Distributions: ContinuousUnivariateDistribution, Exponential, @check_args,
                      Sampleable, Univariate, Continuous,
                      MixtureModel, sampler, DiscreteNonParametricSampler, AliasTable
import Distributions: pdf, logpdf, cdf, quantile, minimum, maximum, mean, var, mgf, cf,
                      insupport

import LinearAlgebra: diag, I, inv
import Random: AbstractRNG, rand

include("phasetype.jl")
include("coxian.jl")
include("hypoexponential.jl")
include("hyperexponential.jl")

end # module PhaseTypeDistributions
