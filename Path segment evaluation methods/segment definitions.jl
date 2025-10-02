#PDMPs are defined by evolving according to some dynamic along a segment (or instantaneously at events). 
#There are many possible dynamics, but each path segment (and every event) follows one or another.
abstract type DynType
end

#If we are done with evolution then it is sometimes useful to set the new type to be a special type.
struct Terminal<:DynType
end

#We can update positions...
abstract type PositionDyn<:DynType
end

#...and we can update velocities
abstract type VelocityDyn<:DynType
end


#A large swath of dynamics are position-velocity dynamics, for which we update the position x according to dx = vdt.
#For now this is the only position dynamic we consider.
#In principle we could consider HMC where the DynType could be PositionMomentum instead.
struct PositionVelocity<:PositionDyn
end

#Velocity updates come in many forms. Here we assume the velocities are updated according to either a gradient reflection or a velocity ODE.
#In principle there may be many forms of gradient reflection updates for a single PDMP method.
#If we encounter such a PDMP we shall generalize to have the parameter N (an integer) in the struct
struct VelocityODE{K<:PDMP_Method}<:VelocityDyn
end

#Gradient reflection generalize the BPS jump kernel.
struct GradientReflection{K<:PDMP_Method}<:VelocityDyn 
end


#Every path segment needs to encode a few bits of information to compute an acceptance rate factor (ARF).
#The rates are given as a (mutable) static array
@kwdef mutable struct PathSegmentValues{K<:PDMP_Method, N <: Integer}
    dyn::DynType
    time::Float64 = 0.0
    forward_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
    reverse_rates::MVector{N, Float64} = zeros(MVector{N, Float64})
    forward_rate_integral::Float64 = 0.0
    reverse_rate_integral::Float64 = 0.0
    terminal::Bool = false
end

include("Instant velocity methods/gradient reflection.jl")
include("ODE-based velocity methods/velocity ODE.jl")
include("Velocity-based position methods/evaluation.jl")