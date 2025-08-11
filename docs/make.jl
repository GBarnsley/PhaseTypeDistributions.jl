using Documenter, PhaseTypeDistributions

makedocs(
    sitename="Phase Type Distributions",
    pages    = [
        "index.md"
    ]
)

deploydocs(
    repo = "github.com/GBarnsley/PhaseTypeDistributions.jl.git",
)