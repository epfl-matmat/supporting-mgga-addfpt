# Energy-only variant of config.jl.
include(joinpath(@__DIR__, "config.jl"))
base_config = get_config()
get_config() = (; base_config..., loss_weights=(; E=10.0, ρ=0.0, τ=0.0))
