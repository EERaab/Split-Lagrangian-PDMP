
"""
    new_point!(pdmp::PDMP, rev_pdmp::PDMP, state::BinaryState, evolution_data::EvolutionData, numerics::NumericalParameters, max_time::Float64, save_skeleton::Bool = false)

Returns a new point for a PDMP together with its acceptance (to be used in the Metropolis correction).
"""
function new_point!(pdmp::PDMP, rev_pdmp::PDMP, state::BinaryState, evolution_data::EvolutionData, numerics::NumericalParameters, max_time::Float64, save_skeleton::Bool = false)
    #We start our "clock" at t = 0 and terminate at t = max_time
    time = 0.0

    #We decide whether to follow the PDMP or its reversal
    pdmp_reversed = rand(Bool)
    if pdmp_reversed
        pdmp_actual = rev_pdmp
    else
        pdmp_actual = pdmp
    end
    #We compute the initial state acceptance and the exponent term
    acceptance = 1.0/full_density_kernel!(pdmp, state, evolution_data, numerics)
    exponent = 0.0
    
    #The initial and final adjustments to acceptances are a bit special and must be treated slightly differently. 
    initial = true
    final = false

    #We start off evolving position, always.
    current_dynamic = PositionVelocity()

    #We shall need an "old" dynamic as well for the reverse rate computations.
    old_dynamic = PositionVelocity()
    skeleton = []
    if save_skeleton
        push!(skeleton, copy.(state.position))
    end

    n_evt = 0

    #We proceed to evaluate a segment at a time until we reach the max time. 
    while time < max_time
        #A segment is computed. Each segment update will conclude by adjusting the dynamic.
        #If the segment is of a position type we update the time
        
        segment, new_dynamic = evaluate_segment!(pdmp_actual, state, evolution_data, numerics, current_dynamic, max_duration = max_time - time)
        n_evt += 1
        if current_dynamic isa PositionDyn 
            time += segment.time
        end
        #We adjust the acceptances and the exponential factor. The final and initial updates are a bit special.
        acceptance *= reverse_acceptance_rate_factor(segment, old_dynamic, initial)
        
        if new_dynamic isa Terminal
            # x is of type Terminal
            #println("terminal time = $time, max_time = $max_time, " ,time < max_time)
            #terminal_reached = true

            time += 0.1 #let's avoid strange things
        end

        if initial
            initial = !initial
        end

        if time ≥ max_time
            final = true
        end

        acceptance *= 1/forward_acceptance_rate_factor(segment, new_dynamic, final)
        exponent += segment.forward_rate_integral - segment.reverse_rate_integral

        #Due to how velocities are handled in some methods we will get NaN-acceptances. 
        if isnan(acceptance)
            #println("Acceptance is NaN!")
            return acceptance, n_evt
        end
        old_dynamic = current_dynamic
        current_dynamic = new_dynamic
        if save_skeleton
            push!(skeleton, copy.(state.position))
        end
    end

    #In the final state we include the density of the final state in our acceptance, as well as the exponential factor
    acceptance *= full_density_kernel!(pdmp, state, evolution_data, numerics)*exp(exponent)

    #The state has been updated as we've  run this function, and is included in the input. Hence we only need to return the acceptance rate
    if save_skeleton
        return acceptance, skeleton
    end
    return acceptance, n_evt
end 


"""
    algorithm(
    pdmp::PDMP,
    nums::NumericalParameters;
    point_number::Integer = 10,
    max_time::Float64 = 1.0,                
    max_computation_time::Float64 = Inf,    
    initial_state::Union{Nothing,BinaryState} = nothing,
    use_correction::Bool = true,
)

Implementation of Metropolis-corrected PDMP sampler. Takes a PDMP and several numerical parameters to simulate a PDMP. Outputs the set of accepted states.
"""
function algorithm(
    pdmp::PDMP,
    nums::NumericalParameters;
    point_number::Integer = 10,
    max_time::Float64 = 1.0,                # stochastic integration time
    max_computation_time::Float64 = Inf,    # new wall-clock time limit (seconds)
    initial_state::Union{Nothing,BinaryState} = nothing,
    use_correction::Bool = true,
)

    k = 0
    n_evt = 0
    state_list = BinaryState[]
    acceptances = Float64[]
    rev_pdmp = reverse(pdmp)
    evo_data = initialize_evolution_data(pdmp)

    # Handle initial state
    if initial_state === nothing
        initial_state = initialize_binary_state!(pdmp, evo_data, nums)
    end
    push!(state_list, initial_state)

    # Record start time
    t_start = time()

    # Main loop
    while k < point_number && (time() - t_start) < max_computation_time
        new_state = BinaryState(
            copy.(state_list[end].position),
            sample_auxiliary!(pdmp, state_list[end].position, evo_data, nums),
        )

        acceptance, n_evt_new = new_point!(pdmp, rev_pdmp, new_state, evo_data, nums, max_time)
        acceptance = min(1, acceptance)
        n_evt += n_evt_new

        if isnan(acceptance)
            acceptance = 0
        end
        push!(acceptances, acceptance)

        if use_correction
            if rand() < acceptance
                push!(state_list, new_state)
            else
                push!(state_list, state_list[end]) # reject → repeat last state
            end
        else
            push!(state_list, new_state)
        end

        k += 1
    end

    return state_list, acceptances, n_evt
end
