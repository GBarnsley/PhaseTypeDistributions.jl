module PhaseTypeDistributions

export PhaseType

import Distributions: ContinuousUnivariateDistribution, Exponential, @check_args, Sampleable, Univariate, Continuous, DiscreteNonParametric
import Distributions: pdf, logpdf, cdf, quantile, minimum, maximum, mean, var, mgf, cf, insupport
#import ExponentialUtilities: exponential!, alloc_mem #make an extension so we can use exponential!(S * x) instead of exp(S * x) if you want
import LinearAlgebra: diag, I, inv
import Random: AbstractRNG, rand

include("phasetype.jl")

end # module PhaseTypeDistributions
