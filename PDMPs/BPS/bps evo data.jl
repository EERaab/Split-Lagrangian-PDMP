include("../call_counters.jl")

struct BPSEvoData<:EvolutionData{BPS_Method}
    gradient::Vector{Float64}
end

function initialize_evolution_data(pdmp::PDMP{<:BPS_Method})
    return BPSEvoData(zeros(Float64, pdmp.target.dimension))
end

function fetch_evo_data!(pdmp::PDMP{<:BPS_Method}, evo_data::BPSEvoData, numerics::NumericalParameters, state::BinaryState, dyn_type::DynType)
    #We update the gradient
    global n_call_grad
    global n_call_point
    n_call_point += 1
    n_call_grad += 1
    if numerics.diff_method isa ForwardDer
        ForwardDiff.gradient!(evo_data.gradient, pdmp.target.log_density, state.position)
    elseif numerics.diff_method isa AnalyticalDer
        numerics.diff_method.gradient!(evo_data.gradient, state.position)
    else
        error("Unspecified derivative method used.")
    end
    nothing
end
