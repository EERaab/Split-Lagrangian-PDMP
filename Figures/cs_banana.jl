include("optim_params.jl")
using Statistics

@everywhere begin
    include("convergence_speed.jl")

    function get_starting_sample()
        return [0.00001,0.00001]
    end

    target = target_rosenbrock
    x_CDF = rosenbrock_CDF
end




function generate_figure_banana_data(max_time, n_repeats, opt_params,overwrite = false)

    h = 0.001

    tol = opt_params["CA-BPS"][2]
    pdmp = PDMP(Version6_2(10000), target, false)
    nums = Version6_2Numerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000))

    #h = 0.5
    tol = opt_params["SL-PDMP"][2]
    pdmp_L = PDMP(Lagrangian(10000), target, false)
    nums_L = LagrangianNumerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000))

    tol = opt_params["BPS"][2]
    pdmp_BPS = PDMP(BPS(), target, false)
    nums_BPS = BPSNumerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000));

    runs_list = [(pdmp,nums,opt_params["CA-BPS"][1],max_time,"CA-BPS"),
                (pdmp_L,nums_L,opt_params["SL-PDMP"][1],max_time,"SL-PDMP"),
                (pdmp_BPS,nums_BPS,opt_params["BPS"][1],max_time,"BPS"),
                ]

    filename = "banana.csv"
    if isfile(filename) & !overwrite
        full_df = CSV.read(filename,DataFrame)
    else
        full_df = generate_samples(runs_list,n_repeats,get_starting_sample)
        CSV.write(filename, full_df)
    end
    df_avg = combine(groupby(full_df, :name), :acceptance => mean => :avg_acceptance)
    print(df_avg)
    #res = plot_ks_evolution(full_df,x_CDF,"Kolmogorov-Smirnov distance , 20-d Gaussian")
    #print("res",res)
    #ggplotres_evo(res,"20d non isotropic gaussian","Number of steps",outfile = "gaussian-20d-ks-evo.png")
    #savefig("gaussian-20d-ks-evt.png")
    #res = plot_ks_evolution_time(full_df,x_CDF,"20d non isotropic gaussian")
    #ggplotres_evo(res,"20d non isotropic gaussian","Time (s)",outfile = "gaussian-20d-ks-evo-time.png")
    #res = plot_ks_evolution_events(full_df,x_CDF,"Kolmogorov-Smirnov distance, 20-d Gaussian")
    #ggplotres_evo(res,"20d non isotropic gaussian","Number of events",outfile = "gaussian-20d-ks-evo-evt.png")
    #savefig("gaussian-20d-ks-time.png")
    return full_df
end 



print(ARGS)
# ARGS is a Vector{String}
if length(ARGS) < 1
    println("Usage: julia myscript.jl <name> <number>")
    exit(1)
end

#name = ARGS[1]
n    = parse(Int, ARGS[1])


if n == -1
    for i in 1:100
        pdmp = PDMP(Version6_2(10000), deepcopy(target), false)
        tol = 0.04
        nums = Version6_2Numerics(position_method = VTAdaptivePiecewiseConstant(tol,0.001,10000));

        evo_data = initialize_evolution_data(pdmp)
        state = initialize_binary_state!(pdmp, evo_data, nums, initial_position = get_starting_sample());
        #println("start")
        #short run to precompile
        states, acceptances, n_evt = algorithm(pdmp, nums,
                                initial_state = state,
                                point_number = 100000,
                                use_correction = true,
                                max_time = 0.04,
                                max_computation_time = 10.)
    end
end

if n <= 3
    PDMPs = [(BPS(),BPSNumerics,"BPS"), (Version6_2(10000),Version6_2Numerics,"CA-BPS"), (Lagrangian(10000),LagrangianNumerics,"SL-PDMP")]
    a,b,name = PDMPs[n]
    filename = "bananaparams" * name * ".txt"
    #First find optimal parameters
    sim_t = 10.
    n_iter = 10
    n_repeats = 100
    opt_params = get_optimal_params(target,x_CDF,get_starting_sample,sim_t,n_iter,n_repeats,[PDMPs[n]],filename)
    print(opt_params)
else
    function read_final_params(filename::AbstractString)
        params = nothing

        open(filename, "r") do io
            for line in eachline(io)
                if startswith(line, "Final parameters =")
                    vals_str = strip(replace(line, "Final parameters =" => ""))
                    nums = parse.(Float64, split(vals_str, r"[,\s\[\]]+"; keepempty=false))
                    params = (nums[1], nums[2])
                    break
                end
            end
        end

        return params
    end
    opt_params = Dict{String,Tuple{Float64,Float64}}()
    opt_params["BPS"]     = read_final_params("bananaparamsBPS.txt")
    opt_params["CA-BPS"]  = read_final_params("bananaparamsCA-BPS.txt")
    opt_params["SL-PDMP"] = read_final_params("bananaparamsSL-PDMP.txt")
    print(opt_params)
    full_df = generate_figure_banana_data(10. ,100,opt_params,true)
    #using DataFrames, CSV
    #filename = "gaussian-d20.csv"
    #full_df = CSV.read(filename,DataFrame)
    add_ks!(full_df, x_CDF)

    df = select(full_df, ([:dim1, :dim2, :iteration, :t, :n_evt, :ks, :name, :repeat, :T, :e, :n_call, :acceptance]))
    df_filtered = filter(:ks => x -> x != -1, df)
    CSV.write("banana-ks.csv", df_filtered)




    # Step 2: group and summarize
    df_summary = combine(groupby(df_filtered, [:name, :iteration]) ,
        :ks => median => :ks_median,
        :t  => median => :t_median,
        :e  => median => :e_median
    )

    using Plots
    # optionally, for nicer plotting with DataFrames:
    # using StatsPlots

    # df_summary has columns: :name, :ks_median, :t_median

    # Basic scatter plot with different colors per name
    plt = scatter(
        df_summary.t_median,          # x-axis
        df_summary.ks_median,         # y-axis
        group = df_summary.name,      # color by method / name
        xlabel = "Median time (t)",
        ylabel = "Median KS (ks)",
        title = "KS vs Time per Method",
        legend = :topright,
        markersize = 6,
        xaxis = :log10,             # log scale for x
        yaxis = :log10              # log scale for y
    )

    savefig(plt,"banana.png")
    display(plt)

    plt = scatter(
        df_summary.iteration,          # x-axis
        df_summary.ks_median,         # y-axis
        group = df_summary.name,      # color by method / name
        xlabel = "Median time (t)",
        ylabel = "Median KS (ks)",
        title = "KS vs Time per Method",
        legend = :topright,
        markersize = 6,
        xaxis = :log10,             # log scale for x
        yaxis = :log10              # log scale for y
    )

    savefig(plt,"banana2.png")
end