include("../../call_counters.jl")




function fetch_point_data!(point_data::PointData, pdmp::PDMP{<:Lagrangian_Method}, state::BinaryState, diff_method::ForwardDer, dyn_type::PositionVelocity)
    global n_call_density
    global n_call_grad
    global n_call_hess
    global n_call_dir_3
    global n_call_3
    global n_call_point
    global time_density
    global time_grad
    global time_hess
    global time_dir_3
    global time_3
    n_call_point += 1
    n_call_density += 1
    n_call_hess += 1
    n_call_dir_3 += 1
    n_call_grad += 1
    #We introduce a shorthand
    t1 = time()
    log_density = pdmp.target.log_density

    #We define the directional hessian function t ↦ Hf(x+tv)
    t2 = time()
    hf = t -> ForwardDiff.hessian(log_density, (state.position .+ (t.*state.auxiliary)))

    #We take a directional derivative on the hessian function and store the value (hessian) and the derivative.
    point_data.position_update_data = ForwardDiff.derivative!(point_data.position_update_data, hf, 0.0)

    #We compute the gradient.
    t3 = time()
    ForwardDiff.gradient!(point_data.gradient, log_density, state.position)
    t4 = time()

    time_density += t2 - t1
    time_dir_3 += t3 - t2
    time_grad += t4 - t3
    nothing
end

function fetch_point_data!(point_data::PointData, pdmp::PDMP{<:Lagrangian_Method}, state::BinaryState, diff_method::AnalyticalDer, dyn_type::PositionVelocity)
    diff_method.gradient!(point_data.gradient, state.position)

    diff_method.hessian!(point_data.position_update_data.value, state.position)
    
    diff_method.third_order_directional!(point_data.position_update_data.derivs[1], state.position, state.auxiliary)
    
    nothing
end
#=
function fetch_point_data!(point_data::PointData, pdmp::PDMP{<:Lagrangian_Method}, state::BinaryState, dyn_type::Union{LagrangianVelocity, BPSLagrangianVelocity, GradientReflection})
    global n_call_density
    global n_call_grad
    global n_call_hess
    global n_call_dir_3
    global n_call_3
    global time_density
    global time_grad
    global time_hess
    global time_dir_3
    global time_3
    n_call_density += 1
    n_call_hess += 1
    n_call_3 += 1
    n_call_grad += 1
    #We introduce a shorthand
    t1 = time()
    log_density = pdmp.target.log_density
    
    #We define the hessian function x ↦ Hf(x)
    t2 = time()
    hf = u -> ForwardDiff.hessian(log_density, u)
    
    #We take a derivative on the hessian function and store the value (hessian) and the derivative.
    point_data.velocity_update_data = ForwardDiff.jacobian!(point_data.velocity_update_data, hf, state.position)

    #We compute the gradient.
    t3 = time()
    if !(dyn_type isa BPSLagrangianVelocity) 
        ForwardDiff.gradient!(point_data.gradient, log_density, state.position)
    end
    t4 = time()

    time_density += t2 - t1
    time_3 += t3 - t2
    time_grad += t4 - t3
    nothing
end
=#

function fetch_hessian!(point_data::PointData, log_density::Function, position::Array{Float64,1}, diff_method::ForwardDer)
    global n_call_hess
    global time_hess

    n_call_hess += 1
    t1 = time()
    point_data.position_update_data.value = ForwardDiff.hessian!(point_data.position_update_data.value, log_density, position)
    t2 = time()
    time_hess = t2 - t1
    nothing
end

function fetch_hessian!(point_data::PointData, log_density::Function, position::Array{Float64,1}, diff_method::AnalyticalDer)
    #global n_call_hess
    #global time_hess

    #n_call_hess += 1
    #t1 = time()
    diff_method.hessian!(point_data.position_update_data.value, position)
    #t2 = time()
    #time_hess = t2 - t1
    nothing
end

function initialize_point_data(dim::Integer)::PointData
    return PointData(zeros(Float64, dim), DiffResults.DiffResult(zeros(Float64, dim, dim),zeros(Float64, dim, dim)), DiffResults.DiffResult(zeros(Float64, dim, dim),zeros(Float64, dim, dim, dim)))
end
