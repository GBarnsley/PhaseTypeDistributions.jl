module PhaseTypeDistributions

export PhaseType

import Distributions: ContinuousUnivariateDistribution, Exponential, @check_args, Sampleable, Univariate, Continuous, DiscreteNonParametric
#import ExponentialUtilities: exponential!, alloc_mem #make an extension so we can use exponential!(S * x) instead of exp(S * x) if you want
import LinearAlgebra: diag, I
import Random: AbstractRNG, rand

include("phasetype.jl")

end # module PhaseTypeDistributions
