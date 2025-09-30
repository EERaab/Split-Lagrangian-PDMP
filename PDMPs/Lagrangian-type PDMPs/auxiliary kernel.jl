
#All Lagrangian-type methods have the same (conditional) density kernel.
#This allows us to define broad methods to sample the velocity or determine the full density kernel for all such PDMPs 
function velocity_covariance_matrix!(pdmp::PDMP{<:Lagrangian_Method}, position::Vector{Float64}, evo_data::LagrangianEvoData, numerics::NumericalParameters)
    #We determine the hessian
    fetch_hessian!(evo_data.point_data, pdmp.target.log_density, position, numerics.diff_method)
    
    #We determine the spectral data associated to the hessian. It has the unfortunate name evo_data.point_data.position_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, evo_data.point_data.position_update_data.value, pdmp.method.hardness)

    covariance_matrix = Symmetric((evo_data.spectral_data.Q)*(evo_data.spectral_data.Dinv)*(evo_data.spectral_data.Q'))
    return covariance_matrix
end

function auxiliary_kernel!(pdmp::PDMP{<:Lagrangian_Method}, state::BinaryState, evo_data::LagrangianEvoData, numerics::NumericalParameters)
    #We determine the hessian
    fetch_hessian!(evo_data.point_data, pdmp.target.log_density, state.position, numerics.diff_method)
    
    #We determine the spectral data associated to the hessian. It has the unfortunate name evo_data.point_data.position_update_data.value
    hessian = evo_data.point_data.position_update_data.value
    fetch_spectral_data!(evo_data.spectral_data, hessian, pdmp.method.hardness)

    Dinv = evo_data.spectral_data.Dinv
    L = evo_data.spectral_data.Q'*state.auxiliary

    return exp(-dot(L, inv(Dinv), L)/2)/sqrt(abs(det(Dinv)))
    #L = evo_data.spectral_data.Q'*state.auxiliary

    #return exp(-(L'*inv(evo_data.spectral_data.Dinv)*L)/2)/sqrt(abs(det(evo_data.spectral_data.Dinv)))
end

