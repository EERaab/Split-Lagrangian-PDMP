function sample_auxiliary!(pdmp::PDMP, position::Array{Float64,1}, evolution_data::EvolutionData, numerics::NumericalParameters)
    #We determine the covariance matrix for the aux distribution of the given PDMP
    cov = velocity_covariance_matrix!(pdmp, position, evolution_data, numerics)

    #The conditional aux distribution is v|x ∼ N(0, cov)
    return rand(MvNormal(cov))
end

function full_density_kernel!(pdmp::PDMP, state::BinaryState, evolution_data::EvolutionData, numerics::NumericalParameters)
    return exp(pdmp.target.log_density(state.position))*auxiliary_kernel!(pdmp, state, evolution_data, numerics)
end
