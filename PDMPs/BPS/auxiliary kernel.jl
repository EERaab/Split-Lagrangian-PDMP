function velocity_covariance_matrix!(pdmp::PDMP{<:BPS_Method}, position::Vector{Float64}, evo_data::BPSEvoData, numerics::BPSNumerics)
    return I
end

function auxiliary_kernel!(pdmp::PDMP{<:BPS_Method}, state::BinaryState, evo_data::BPSEvoData, numerics::BPSNumerics)
    #Some constant factor is missing here. It will not matter.
    return exp(-dot(state.auxiliary, state.auxiliary)/2)
end