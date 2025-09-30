using Distributed

ncpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))

# Add one worker per CPU
addprocs(ncpus; exeflags="--threads=1")   # each worker single-threaded

@everywhere begin
    include("../main.jl")
    include("ks_statistic")

    #using Base.Threads
    using BlackBoxOptim
    using DifferentialEquations
    using Optim
end

"""
dim = 20
eps = 0.01
diag_cov = vcat(1.0/eps, fill(eps, dim - 1))
function log_pdf_diag_gaussian(x::Vector)
    quad_term = -0.5 * sum(x.^2 ./ diag_cov)
    return quad_term
end
x_CDF = x -> cdf(Normal(0,1/sqrt(eps)), x)


function get_starting_sample()
    return randn(dim) .* diag_cov
end

target = TargetData(log_pdf_diag_gaussian, dim)
"""


function get_optimal_params(target,x_CDF,get_starting_sample,max_time_sim,n_iter,num_repeats = 100, 
    PDMPs = [(BPS(),BPSNumerics,"BPS"), (Version6_2(10000),Version6_2Numerics,"CA-BPS"), (Lagrangian(10000),LagrangianNumerics,"SL-PDMP")],
    filename = "result.txt")

    #PDMPs = [(BPS(),BPSNumerics,"BPS"), (Version6_2(10000),Version6_2Numerics,"CA-BPS"), (Lagrangian(10000),LagrangianNumerics,"SL-PDMP")]
    #PDMPs = [(Lagrangian,LagrangianNumerics,"SL-PDMP")]

    opt_params = Dict{String,Tuple{Float64,Float64}}()

    for (pdmp_type, nums_type, name) in PDMPs
        println("Finding optimal params for ", name)
        #T = 1.0
        #tol = .1
        h = 0.001

        function f(T,tol,pdmp_type_instanciated,nums_type)

            pdmp = PDMP(pdmp_type_instanciated, deepcopy(target), false)
            nums = nums_type(position_method = VTAdaptivePiecewiseConstant(tol,h,10000));

            evo_data = initialize_evolution_data(pdmp)
            state = initialize_binary_state!(pdmp, evo_data, nums, initial_position = get_starting_sample());
            #println("start")
            #short run to precompile
            algorithm(pdmp, nums,
                                    initial_state = state,
                                    point_number = 1000000,
                                    use_correction = true,
                                    max_time = T/10,
                                    max_computation_time = max_time_sim)
            elapsed = @elapsed begin
                states, acceptances, n_evt = algorithm(pdmp, nums,
                                    initial_state = state,
                                    point_number = 1000000,
                                    use_correction = true,
                                    max_time = T,
                                    max_computation_time = max_time_sim)
            end
            pos = [s.position[1] for s in states]
            #println(" ",length(pos))
            println("Time: ",elapsed)
            #print(" ",mean(acceptances))

            err = ks_statistic(pos,x_CDF)
            return err

        end

        function fn(x)
            println("Trying x = ", x, "\n")
            T = exp(x[1])
            tol = exp(x[2])
            #println(T," ",tol)
            errs = zeros(Float64, num_repeats)
            #num_repeats = 8 #@threads
            #@threads for r in 1:num_repeats 
            errs = pmap(r -> f(T,tol,pdmp_type,nums_type), 1:num_repeats)
            print(x, " " ,T," ",tol," stdev",std(errs)," median ",median(errs),"\n")
            return median(errs)
        end


        function gfn(x)
            fxs = zeros(3)
            h=3e-1
            # points to evaluate: baseline, perturb x1, perturb x2
            pts = [
                x,
                [x[1] + h, x[2]],
                [x[1], x[2] + h]
            ]

            # parallel evaluation of fn
            #@threads for j in 1:3
            for j in 1:3
                fxs[j] = fn(pts[j])
            end

            fx = fxs[1]
            grad = [(fxs[2] - fx) / h, (fxs[3] - fx) / h]

            println("x ",x," fx ",fx," grad ",grad,"\n")
            return fx, grad
        end
        function fg!(F, G, x)
            fx, grad = gfn(x)
            if G !== nothing
                G[:] = grad       # write gradient vector into G
            end
            if F !== nothing
                F = fx          # write scalar value into F
                return fx
            end
        end

        #actual optimization
        """
        x0 = [log(1.0),log(0.1)]
        #res = optimize(Optim.only_fg!(fg!), x0, GradientDescent(), Optim.Options(iterations=10))
        for i in 1:30
            fx,gx = gfn(x0)
            println("Iter ",i," f(x) = ",fx," g(x) = ",gx)
            if fx > 0.999
                x0 += [0.,-0.5]
            else
                x0 -= 0.1 .* gx ./ norm(gx) # gradient step with step size 0.5  
            end
        end"""

        #println("Converged: ", Optim.converged(res))
        #println("Minimizer: ", Optim.minimizer(res))
        #println("Minimum value: ", Optim.minimum(res))
        #final_fit = fn(x0)#Optim.minimum(res)
        #final_params = x0#Optim.minimizer(res)
        # Example function f: R^2 -> [0,1]

        # Known intervals for x1 and x2
        x1_bounds = (log(1e-3), log(2e1))
        x2_bounds = (log(1e-3), log(1e1))

        # Inner minimization: given x1, minimize over x2
        function g(x1::Float64)
            # Define 1D function of x2
            f_x2(x2::Float64) = fn([x1, x2])

            # Optimize using Brent method with a 60s time limit
            res = optimize(
                f_x2,
                x2_bounds[1], x2_bounds[2],
                Optim.Brent(),
                iterations = n_iter
            )
            println("inner ", Optim.minimizer(res)," ",Optim.minimum(res))
            return Optim.minimum(res)  # minimum over x2
        end

        # Outer minimization over x1
        res_outer = optimize(
            g,
            x1_bounds[1], x1_bounds[2],
            Optim.Brent(),
            iterations = n_iter
        )

        res = Optim.minimizer(res_outer)
        x1_opt = Optim.minimizer(res_outer)
        f1_opt = Optim.minimum(res_outer)
        # Given optimal x1, find optimal x2
        res = optimize(x -> fn([x1_opt, x]), x2_bounds[1], x2_bounds[2], Optim.Brent(), iterations = n_iter)
        x2_opt = Optim.minimizer(res)
        f2_opt = Optim.minimum(res)
        
        println("Optimal x1 = ", x1_opt, " ", f1_opt)
        println("Optimal x2 = ", x2_opt, " ", f2_opt)
        final_fit = f2_opt
        final_params = [x1_opt, x2_opt]
        #res = bboptimize(fn; SearchRange = [(log(1e-3), log(2e1)), (log(1e-3), log(1e1))],MaxTime = max_time_optim)
        #final_fit = best_fitness(res)
        #final_params = best_candidate(res)

        open(filename, "a") do io
            println(io, "Results for ", name)
            println(io, "Final fitness = ", final_fit)
            println(io, "Final parameters = ", [exp(final_params[1]), exp(final_params[2])])
        end
        opt_params[name] = (exp(final_params[1]), exp(final_params[2]))
    end

    return opt_params
end

#get_optimal_params(target,x_CDF,get_starting_sample,10.,60. * 2,32,4)