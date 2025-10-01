using LinearAlgebra, StaticArrays, DiffResults, ForwardDiff, OrdinaryDiffEq, Distributions, TensorOperations, Plots, Random, FastLapackInterface
#using BenchmarkTools

include("pdmp definitions.jl")
include("kernels and sampling.jl")
include("algorithm.jl")
include("Path segment evaluation methods/segment definitions.jl")
include("PDMPs/Lagrangian-type PDMPs/lagrangian-type.jl")
include("PDMPs/BPS/bps-type.jl")
include("Visualization/plotting.jl")