#The other subtype, apart from the Lagrangian type is just BPS, with a single pdmp type. 
struct BPS<:BPS_Method
end

@kwdef struct BPSNumerics<:NumericalParameters{BPS_Method}
    #position_type::Type = Vector{Float64}
    #auxiliary_type::Type = Vector{Float64}
    position_method::PositionMethod = VTPiecewiseConstant(0.01)
    diff_method::DifferentiationMethod = ForwardDer()
end

function select_dynamic(segment::PathSegmentValues{BPS, N}) where N
    if segment.dyn isa GradientReflection
        return PositionVelocity()
    else
        return GradientReflection{BPS}()
    end
end

function reverse_acceptance_rate_factor(segment::PathSegmentValues{BPS, N}, old_dynamic::DynType, initial::Bool) where N
    if initial || (segment.dyn isa GradientReflection)
        return 1.0
    end
    if segment.dyn isa PositionVelocity
        return segment.reverse_rates[1]
    end
    error("Incorrect dynamic selected. Old: "*string(old_dynamic)*" and current: "*string(segment.dyn))
end

function forward_acceptance_rate_factor(segment::PathSegmentValues{BPS, N}, new_dynamic::DynType, final::Bool) where N
    if final || (segment.dyn isa GradientReflection{BPS})
        return 1.0
    end
    if segment.dyn isa PositionVelocity
        return segment.forward_rates[1]
    end
    error("Incorrect dynamic selected. New: "*string(new_dynamic)*" and current: "*string(segment.dyn))
end


function rate_number(pdmp::PDMP{BPS}, dyn::DynType)::Int
    return 1
end
