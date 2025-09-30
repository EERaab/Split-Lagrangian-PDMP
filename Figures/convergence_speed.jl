#########################################################################################################
#########################################################################################################
# This file: generate kolmogorv smirnov distance evolution over time, looking at the 1rst coordinate only
# for several different targets. The parameters (namely T and tol) are hand tuned for a fair comparison.
#########################################################################################################
#########################################################################################################
include("../main.jl")
include("ks_statistic")
include("../Tests/Twin peaks.jl");
target_list = Dict("Twin peaks" => target)
include("../Tests/Banana.jl")
#target_list["Banana"] = target

using LinearAlgebra
using Plots
using Measures
using StatsBase, Statistics, Plots
using CSV
using DataFrames, Distributed
using ColorSchemes
using RCall

"""
function ks_statistic(data, cdf)
    sorted_data = sort(data)
    n = length(data)
    D = 0.0
    for (i, x) in enumerate(sorted_data)
        Fn_minus = (i - 1) / n
        Fn_plus  = i / n
        F = cdf(x)
        d = max(abs(F - Fn_minus), abs(F - Fn_plus))
        D = max(D, d)
    end
    return D
end
"""


function add_ks!(df, ref; nchecks=100)
    n = nrow(df)
    df.ks        = Vector{Float64}(undef, n)
    df.iteration = Vector{Int}(undef, n)
    df.t         = Vector{Float64}(undef, n)
    df.e         = Vector{Float64}(undef, n)
    df.n_call    = Float64.(df.n_call)  # Ensure n_call is Float64

    gdf = groupby(df, [:name, :repeat])
    for sub in gdf
        samples = Float64[]
        elapsed = first(sub.elapsed)
        n_evt   = first(sub.n_evt)
        n_call   = first(sub.n_call)
        m = length(sub.dim1)                    # number of samples in this group

        # choose ~nchecks evenly spaced indices (not exceeding m)
        check_indices = round.(Int, range(1, m; length=min(nchecks, m)))

        for (j, _) in enumerate(eachindex(sub.dim1))
            global_index = parentindices(sub)[1][j]

            # accumulate sample
            push!(samples, sub.dim1[j])

            # iteration / normalized "time"
            df.iteration[global_index] = j
            df.t[global_index] = elapsed * j / m
            df.e[global_index] = n_evt   * j / m
            df.n_call[global_index] = n_call  * j / m

            # KS at checkpoints only
            if j in check_indices
                df.ks[global_index] = ks_statistic(samples, ref)
            else
                df.ks[global_index] = -1.0
            end
        end
    end

    return df
end

function do_run(pdmp,nums,max_time,T,get_starting_sample)
    global n_call_point
    evo_data = initialize_evolution_data(pdmp)
    state = initialize_binary_state!(pdmp, evo_data, nums, initial_position = get_starting_sample());
    reset_call_counters()
    println("start")
    states, acceptances, n_evt = algorithm(pdmp, nums,
                        initial_state = state,
                        point_number = 10000000000,
                        use_correction = true,
                        max_time = T,
                        max_computation_time = max_time)
    return [s.position for s in states], mean(acceptances), n_evt, n_call_point
end



function generate_samples(runs_list, num_repeats, get_starting_sample::Function)
    all_dfs = DataFrame[]

    for (pdmp, nums, T, max_time, name) in runs_list
        println("\nT=", T, " name = $name")

        # Each repeat is independent, so distribute them
        partial_results = pmap(1:num_repeats) do r
            samples_, acc, n_evt, n_call_point = do_run(pdmp, nums, 60., T, get_starting_sample) #short run to precompile things
            elapsed = @elapsed begin
                samples_, acc, n_evt, n_call_point = do_run(pdmp, nums, max_time, T, get_starting_sample)
            end
            println(elapsed, " " , length(samples_))

            df = DataFrame([getindex.(samples_, i) for i in 1:length(samples_[1])],
                           Symbol.("dim", 1:length(samples_[1])))

            df.repeat     = fill(r, size(df, 1))
            df.name       = fill(name, size(df, 1))
            df.acceptance = fill(acc, size(df, 1))
            df.elapsed    = fill(elapsed, size(df, 1))
            df.N          = fill(length(samples_), size(df, 1))
            df.T          = fill(T, size(df, 1))
            df.n_evt      = fill(n_evt, size(df, 1))
            df.n_call     = fill(n_call_point, size(df, 1))

            return df
        end

        run_df = vcat(partial_results...)
        push!(all_dfs, run_df)
    end

    return vcat(all_dfs...)
end
### now plot


function plot_ks_evolution(full_df,x_CDF,title,uplow=false)

    names = unique(full_df.name)
    colors = distinguishable_colors(length(names))[1:length(names)]#ColorSchemes.tab10.colors[1:length(names)]

    plt = plot(
        xlabel = "Sample size", ylabel = "KS statistic",
        title = title,
        xscale = :log10, yscale = :log10,
        legend = :bottomleft,legend_background_color = :transparent,
        size=(2000, 1000)
    )
    results = DataFrame()
    for (idx, name) in enumerate(names)
        c = colors[idx]
        subdf = filter(row -> row.name == name, full_df)
        grouped = groupby(subdf, :repeat)
        all_x_samples = [group.dim1 for group in grouped]

        min_samples = 20
        max_samples = subdf.N[1]
        sizes = unique(round.(Int, 10 .^ range(log10(min_samples), log10(max_samples), length = 100)))


        ks_matrix = fill(NaN, length(all_x_samples), length(sizes))
        for (r, x_samples) in enumerate(all_x_samples)
            for (j, s) in enumerate(sizes)
                if s <= length(x_samples)
                    ks_matrix[r, j] = ks_statistic(x_samples[1:s], x_CDF)
                end
            end
        end

        mean_ks = mapslices(x -> mean(skipmissing(x)), ks_matrix, dims=1)[:]
        low_ks  = mapslices(x -> quantile(skipmissing(x), 0.025), ks_matrix, dims=1)[:]
        high_ks = mapslices(x -> quantile(skipmissing(x), 0.975), ks_matrix, dims=1)[:]
        append!(results, DataFrame(
            sizes = sizes,
            mean_ks = mean_ks,
            low_ks = low_ks,
            high_ks = high_ks,
            name = name
        ))
        plot!(plt, sizes, mean_ks; label = "Mean KS ($name)", lw = 2, color = c)
        if uplow
            plot!(plt, sizes, low_ks; label = "", lw = 1, color = c, linestyle = :dash)
            plot!(plt, sizes, high_ks; label = "", lw = 1, color = c, linestyle = :dash)
        end
    end

    display(plt)
    return results
end

function ggplotres_evo(results, title = "", xlabel  ="", uplow = false; outfile::Union{Nothing,String}=nothing)
    print(xlabel)
    @rput results title uplow outfile xlabel

    R"""
    library(ggplot2)
    library(scales)

    p <- ggplot(results, aes(x = sizes, y = mean_ks, color = name)) +
      geom_line(size = 1) +
      #coord_fixed()+
  #coord_trans(y="log10", x="log10") +
  scale_y_continuous(trans = log10_trans(),
                     breaks = trans_breaks("log10", function(x) 10^x),
                     labels = trans_format("log10", math_format(10^.x))) +
  scale_x_continuous(trans = log10_trans(),
                     breaks = trans_breaks("log10", function(x) 10^x),
                     labels = trans_format("log10", math_format(10^.x)))+
      labs(
        x = xlabel,
        y = "KS distance",
        title = title,
        color = "Name"
      ) +
      theme_minimal(base_size = 16)

    if (uplow) {
      p <- p +
        geom_line(aes(y = low_ks), linetype = "dashed") +
        geom_line(aes(y = high_ks), linetype = "dashed")
    }

    if (is.null(outfile)) {
      pngfile <- tempfile(fileext = ".png")
    } else {
      pngfile <- outfile
    }

    ggsave(filename = pngfile, plot = p, width = 10, height = 5, dpi = 250)
    pngfile
    """

    pngfile = rcopy(R"pngfile")
    display("image/png", read(pngfile))
end

function plot_ks_evolution_time(full_df,x_CDF,title,uplow=false)

    plt_time = plot(
        xlabel = "Estimated time (s)", ylabel = "KS statistic",
        title = title,
        xscale = :log10, yscale = :log10,
        #legend = :topright,
        legend = :bottomleft, legend_background_color = :transparent,
        size=(2000, 1000)
    )

    names = unique(full_df.name)
    colors = distinguishable_colors(length(names)+10)[1:length(names)]#ColorSchemes.tab10.colors[1:length(names)]
    results = DataFrame()
    for (idx, name) in enumerate(names)
        c = colors[idx]
        subdf = filter(row -> row.name == name, full_df)
        grouped = groupby(subdf, :repeat)
        all_x_samples = [group.dim1 for group in grouped]
        elapsed_times = [first(group.elapsed) for group in grouped]  # one per repeat

        min_samples = 20
        max_samples = subdf.N[1]
        sizes = unique(round.(Int, 10 .^ range(log10(min_samples), log10(max_samples), length = 100)))


        ks_matrix = fill(NaN, length(all_x_samples), length(sizes))
        for (r, x_samples) in enumerate(all_x_samples)
            for (j, s) in enumerate(sizes)
                if s <= length(x_samples)
                    ks_matrix[r, j] = ks_statistic(x_samples[1:s], x_CDF)
                end
            end
        end

        mean_ks = mapslices(x -> mean(skipmissing(x)), ks_matrix, dims=1)[:]
        low_ks  = mapslices(x -> quantile(skipmissing(x), 0.025), ks_matrix, dims=1)[:]
        high_ks = mapslices(x -> quantile(skipmissing(x), 0.975), ks_matrix, dims=1)[:]

        avg_elapsed = mean(elapsed_times)  # per name
        print(avg_elapsed)
        estimated_times = avg_elapsed .* sizes ./ max_samples
        append!(results, DataFrame(
            sizes = estimated_times,
            mean_ks = mean_ks,
            low_ks = low_ks,
            high_ks = high_ks,
            name = name
        ))
        plot!(plt_time, estimated_times, mean_ks; label = "Mean KS ($name)", lw = 2, color = c)
        if uplow
            plot!(plt_time, estimated_times, low_ks; label = "", lw = 1, color = c, linestyle = :dash)
            plot!(plt_time, estimated_times, high_ks; label = "", lw = 1, color = c, linestyle = :dash)
        end
    end

    display(plt_time)
    return results
end


function plot_ks_evolution_events(full_df,x_CDF,title,uplow=false)

    plt_time = plot(
        xlabel = "Number of events", ylabel = "KS statistic",
        title = title,
        xscale = :log10, yscale = :log10,
        #legend = :topright,
        legend = :bottomleft, legend_background_color = :transparent,
        size=(2000, 1000)
    )

    names = unique(full_df.name)
    colors = distinguishable_colors(length(names))[1:length(names)]#ColorSchemes.tab10.colors[1:length(names)]
    results = DataFrame()
    for (idx, name) in enumerate(names)
        c = colors[idx]
        subdf = filter(row -> row.name == name, full_df)
        grouped = groupby(subdf, :repeat)
        all_x_samples = [group.dim1 for group in grouped]
        n_events = [first(group.n_evt) for group in grouped]  # one per repeat

        min_samples = 20
        max_samples = subdf.N[1]
        sizes = unique(round.(Int, 10 .^ range(log10(min_samples), log10(max_samples), length = 100)))


        ks_matrix = fill(NaN, length(all_x_samples), length(sizes))
        for (r, x_samples) in enumerate(all_x_samples)
            for (j, s) in enumerate(sizes)
                if s <= length(x_samples)
                    ks_matrix[r, j] = ks_statistic(x_samples[1:s], x_CDF)
                end
            end
        end

        mean_ks = mapslices(x -> mean(skipmissing(x)), ks_matrix, dims=1)[:]
        low_ks  = mapslices(x -> quantile(skipmissing(x), 0.025), ks_matrix, dims=1)[:]
        high_ks = mapslices(x -> quantile(skipmissing(x), 0.975), ks_matrix, dims=1)[:]

        avg_events = mean(n_events)  # per name
        print(n_events)
        estimated_events = avg_events .* sizes ./ max_samples
        append!(results, DataFrame(
            sizes = estimated_events,
            mean_ks = mean_ks,
            low_ks = low_ks,
            high_ks = high_ks,
            name = name
        ))
        plot!(plt_time, estimated_events, mean_ks; label = "Mean KS ($name)", lw = 2, color = c)
        if uplow
            plot!(plt_time, estimated_events, low_ks; label = "", lw = 1, color = c, linestyle = :dash)
            plot!(plt_time, estimated_events, high_ks; label = "", lw = 1, color = c, linestyle = :dash)
        end
    end

    display(plt_time)
    return results
end

###################################################################################################################
###################################################################################################################
###################################################################################################################
# Figure generation
###################################################################################################################
###################################################################################################################
###################################################################################################################

###################################################################################################################
# 20-d gaussian



function sanity_check(full_df,x_CDF)
    #df_first_repeat = filter(row -> row.repeat == 1, full_df)

    grouped = groupby(full_df, :name)

    df_by_name = Dict(name => DataFrame(subdf) for (name, subdf) in pairs(grouped))
    for (name,df) in df_by_name
        p = plot()
        by_r = groupby(df, :repeat)
        for (r, df_r) in Dict(name => DataFrame(subdf) for (name, subdf) in pairs(by_r))
            #print(df)
            x_samples = df_r.dim1
            x_CDF = rosenbrock_CDF
            #theoretical = Normal(0, 1)
            x_vals = range(minimum(x_samples) - 1, stop=maximum(x_samples) + 1, length=500)

            # Step 4: Plot
            empirical_cdf = ecdf(x_samples)
            plot!(x_vals, empirical_cdf.(x_vals);
                label = "Empriical CDF",lw=1, color=:red, title = name)
        end
        x_vals = range(-10,10, length=500)
        theoretical_cdf_vals = x_CDF(x_vals)
        plot!(x_vals, theoretical_cdf_vals;
            label = "Theoretical CDF", lw=2, color=:blue)
        display(p)
    end    
end

###################################################################################################################
# banana distribution


function generate_figure_banana(N = 1000,n_repeats = 6*6,overwrite = false)



    function get_starting_sample()
        return [0.00001,0.00001]
    end

    target = target_rosenbrock
    x_CDF = rosenbrock_CDF

    h = 0.0001
    tol = .7
    pdmp = PDMP(Version6_2(0.1), target, false)
    nums = Version6_2Numerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000))

    #h = 0.5
    tol = 0.07
    pdmp_L = PDMP(Lagrangian(10000.), target, false)
    nums_L = LagrangianNumerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000))

    tol = 0.005
    pdmp_BPS = PDMP(BPS(), target, false)
    nums_BPS = BPSNumerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000));


    #hand tuned params for BPS: T = 10, tol = 0.005
    #hand tuned params for V6: T = 1., tol = 1.
    #runs_list = [(pdmp,nums,"Version 6.2 h=0.5"),(pdmp_L,nums_L,"Lagrangian h=0.5"),(pdmp_BPS,nums_BPS,"BPS h=0.5")]
    #runs_list = [(pdmp,Version6_2Numerics(position_method = VTAdaptivePiecewiseConstant(0.2,h,10000)),10.,100,"V6 T = 10"),
    #            (pdmp,Version6_2Numerics(position_method = VTAdaptivePiecewiseConstant(1.,h,10000)),1.,1000,"V6 T = 1."),
    #            (pdmp,Version6_2Numerics(position_method = VTAdaptivePiecewiseConstant(100.,h,10000)),.1,10000,"V6 T = .1")
    #            ]
    tol_vals = exp10.(range(log10(0.001), log10(.01), length=3))
    T_tot = 1. *N
    T_vals = exp10.(range(log10(10.), log10(20.), length=3))
    print(T_vals)
    runs_list = Any[(pdmp_BPS,BPSNumerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000)),T,Int(round(T_tot / T)),"BPS T = $T. tol=$tol") for tol in tol_vals for T in T_vals]
    append!(runs_list,Any[(pdmp_L,LagrangianNumerics(position_method = VTAdaptivePiecewiseConstant(tol,h,10000)),T,Int(round(T_tot / T)),"Lagrangian T = $T. tol=$tol") for tol in tol_vals for T in T_vals] )
    push!(runs_list,(pdmp_L,LagrangianNumerics(position_method = VTAdaptivePiecewiseConstant(.005,h,10000)),15.,Int(round(T_tot / 15.)),"Lagrangian T = 15. 0.05"))
                #=(pdmp_L,LagrangianNumerics(position_method = VTAdaptivePiecewiseConstant(.04,h,10000)),10.,N,"Lagrangian T = 10. 0.04"),
                (pdmp_L,LagrangianNumerics(position_method = VTAdaptivePiecewiseConstant(.00,h,10000)),10.,N,"Lagrangian T = 10. 0.04"),
                #(pdmp_BPS,BPSNumerics(position_method = VTAdaptivePiecewiseConstant(0.001,h,10000)),10.,N,"BPS T = 10 0.001"),
                #(pdmp_BPS,BPSNumerics(position_method = VTAdaptivePiecewiseConstant(0.01,h,10000)),10.,N,"BPS T = 10 0.001"),
                #(pdmp_BPS,BPSNumerics(position_method = VTAdaptivePiecewiseConstant(0.001,h,10000)),1.,N,"BPS T = 1. 0.001"),
                #(pdmp_BPS,BPSNumerics(position_method = VTAdaptivePiecewiseConstant(0.001,h,10000)),0.1,N,"BPS T = 0.1 0.001"),
                #(pdmp_BPS,BPSNumerics(position_method = VTAdaptivePiecewiseConstant(0.0005,h,10000)),10.,N,"BPS T = 10"),
                ]=#

    filename = "banana.csv"
    if isfile(filename) & !overwrite
        full_df = CSV.read(filename,DataFrame)
    else
        full_df = generate_samples(runs_list,n_repeats,get_starting_sample)
        CSV.write(filename, full_df)
    end
    df_avg = combine(groupby(full_df, :name), :acceptance => mean => :avg_acceptance)
    print(df_avg)
    sanity_check(full_df,x_CDF)
    plot_ks_evolution(full_df,x_CDF,"Kolmogorov-Smirnov distance, banana")
    plot_ks_evolution_time(full_df,x_CDF,"Kolmogorov-Smirnov distance, banana")
    plot_ks_evolution_events(full_df,x_CDF,"Kolmogorov-Smirnov distance, banana")
    #ESS_plot(full_df,"ESS plot")
end 

#generate_figure_banana(10000,24,true)



###################################################################################################################
###################################################################################################################
###################################################################################################################

## V2

# KS Statistic Function

# Parameters
#Plot all empirical cdf
"""
using StatsBase

# Step 3: Theoretical CDF (for Normal(0,1))
if false
    full_df = CSV.read("banana.csv",DataFrame)
    df_first_repeat = filter(row -> row.repeat == 1, full_df)

    grouped = groupby(df_first_repeat, :name)

    df_by_name = Dict(name => DataFrame(subdf) for (name, subdf) in pairs(grouped))
    for (name,df) in df_by_name
        #print(df)
        x_samples = df.dim1
        x_CDF = rosenbrock_CDF
        #theoretical = Normal(0, 1)
        x_vals = range(minimum(x_samples) - 1, stop=maximum(x_samples) + 1, length=500)
        theoretical_cdf_vals = x_CDF(x_vals)

        # Step 4: Plot
        p = plot(x_vals, theoretical_cdf_vals;
        label = "Theoretical CDF", lw=2, color=:blue)
        empirical_cdf = ecdf(x_samples)
        plot!(x_vals, empirical_cdf.(x_vals);
            label = "Empriical CDF",lw=1, color=:red)
        plot!(x_vals, theoretical_cdf_vals;
            label = "Theoretical CDF", lw=2, color=:blue)
        display(p)
    end
end
"""
