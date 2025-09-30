function evaluate_segment!(pdmp::PDMP{T}, state::BinaryState, evo_data::EvolutionData, numerics::NumericalParameters, dyn_type::GradientReflection{T}; max_duration::Float64 = 0.0) where T<:PDMP_Method
    #Because the gradient reflections are deterministic there is no need to contain any information in the path segment. None the less, we include one for compatibility.
    segment = PathSegmentValues{T, 0}(dyn = dyn_type)

    #We get the relevant evolution data
    fetch_evo_data!(pdmp, evo_data, numerics, state, dyn_type)

    #Old version
    #w = reflection_covector(pdmp, evo_data, state, pdmp.method)
    #invG_w = (evo_data.spectral_data.Q)*evo_data.spectral_data.Dinv*(evo_data.spectral_data.Q')*w
    #The reflection is now trivial to compute
    #state.auxiliary .-= 2*invG_w*((state.auxiliary'*w)/(w'*invG_w))
    #
    #

    #We introduce a shorthand definition. 
    reflect!(state, pdmp, evo_data)

    #New dynamics
    new_dyn = select_dynamic(segment)
    return segment, new_dyn
end
