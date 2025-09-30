include("../main.jl")
include("../Tests/Twin peaks.jl");
target_list = Dict("Twin peaks" => target)
include("../Tests/Banana.jl")
target_list["Banana"] = target

using LinearAlgebra
using Plots
using Measures

# === Covariance Ellipse and Axes ===
function covariance_ellipse_and_axes(
    Σ::AbstractMatrix{<:Real},
    center::Vector{<:Real};
    scale=1.0,
    npoints=100
)
    Σ = Matrix(Σ)
    vals, vecs = eigen(Σ)
    a, b = sqrt.(vals) .* scale
    t = range(0, 2π; length=npoints)
    ellipse = [a * cos.(t)'; b * sin.(t)']
    ellipse_rotated = vecs * ellipse .+ center
    axis1 = (center, center .+ vecs[:,1] * a)
    axis2 = (center, center .+ vecs[:,2] * b)
    return ellipse_rotated, axis1, axis2
end

function f_(center,pdmp,evo_data)
    cov = velocity_covariance_matrix!(pdmp, center, evo_data)
end

function visualize_velocity_space(target,xmin,xmax,ymin,ymax,ellipse_grid_size = 10, ellipse_scale = 0.4)
    pdmp = PDMP(LagrangianGradientReflection(), target, false)
    evo_data = initialize_evolution_data(pdmp)

    heatmap_grid_size = 1000
    x_heat = range(xmin, xmax; length=heatmap_grid_size)
    y_heat = range(ymin, ymax; length=heatmap_grid_size)

    x_ellipse = range(xmin, xmax; length=ellipse_grid_size)
    y_ellipse = range(ymin, ymax; length=ellipse_grid_size)

    # === Evaluate g over fine grid ===
    g_values = [exp(target.log_density([x, y])) for y in y_heat, x in x_heat]  # note: row-major (y, x)

    # === Plot ===
    heatmap(
        x_heat, y_heat, g_values;
        c=:grays,
        #aspect_ratio=1,
        xlabel="x", ylabel="y",
        legend=false,
        #title=plot_title,
    )
    plot!(axis=false, legend=false, grid=false, 
    framestyle=:none,margin=0mm,ticks=nothing,    xlims=(xmin, xmax),
    ylims=(ymin, ymax))

    g_vals = [exp(target.log_density([x, y])) for x in x_ellipse, y in y_ellipse]
    g_min, g_max = minimum(g_vals), maximum(g_vals)

    # === Overlay Ellipses and Principal Axes ===
    for x in x_ellipse, y in y_ellipse
        center = [x, y]
        Σ = f_(center,pdmp,evo_data)
        ellipse, axis1, axis2 = covariance_ellipse_and_axes(Σ, center; scale=ellipse_scale)
        g_val = exp(target.log_density([x, y]))
        alpha_val = (g_val - g_min) / (g_max - g_min + 1e-10)*0.7 + 0.3  # avoid div by 0
        # Draw ellipse
        plot!(ellipse[1, :], ellipse[2, :], lw=2, color=RGBA(1, 0, 0, alpha_val))
        scatter!([center[1]], [center[2]]; color=:black, marker=:x, markersize=4*ellipse_scale)
        # Draw principal axes
        #plot!([axis1[1][1], axis1[2][1]], [axis1[1][2], axis1[2][2]], lw=1, color=:red)
        #plot!([axis2[1][1], axis2[2][1]], [axis2[1][2], axis2[2][2]], lw=1, color=:green)
    end

    display(current())
end

function visualize_velocity_space_at_points(target,point_list,xmin,xmax,ymin,ymax,ellipse_grid_size = 10, ellipse_scale = 0.4)
    pdmp = PDMP(LagrangianGradientReflection(), target, false)
    evo_data = initialize_evolution_data(pdmp)

    heatmap_grid_size = 1000
    x_heat = range(xmin, xmax; length=heatmap_grid_size)
    y_heat = range(ymin, ymax; length=heatmap_grid_size)

    # === Evaluate g over fine grid ===
    g_values = [exp(target.log_density([x, y])) for y in y_heat, x in x_heat]  # note: row-major (y, x)

    # === Plot ===
    heatmap(
        x_heat, y_heat, g_values;
        c=:grays,
        #aspect_ratio=1,
        xlabel="x", ylabel="y",
        legend=false,
        padding = (0.0, 0.0),
        #title=plot_title,
    )

    plot!(axis=false, legend=false, grid=false, 
    framestyle=:none,margin=0mm,ticks=nothing,    xlims=(xmin, xmax),
    ylims=(ymin, ymax))
    #plot!(margin=0mm)
    #plot!(left_margin=0mm, bottom_margin=0mm)
    g_vals = [exp(target.log_density(center)) for center in point_list]
    g_min, g_max = minimum(g_vals), maximum(g_vals)

    # === Overlay Ellipses and Principal Axes ===
    for center in point_list
        #center = [x, y]
        Σ = f_(center,pdmp,evo_data)
        ellipse, axis1, axis2 = covariance_ellipse_and_axes(Σ, center; scale=ellipse_scale)
        g_val = exp(target.log_density(center))
        alpha_val = 1#(g_val - g_min) / (g_max - g_min + 1e-10)*0.7 + 0.3  # avoid div by 0
        # Draw ellipse
        plot!(ellipse[1, :], ellipse[2, :], lw=2, color=RGBA(1, 0, 0, alpha_val))
        scatter!([center[1]], [center[2]]; color=:black, marker=:x, markersize=4 * ellipse_scale)
        # Draw principal axes
        #plot!([axis1[1][1], axis1[2][1]], [axis1[1][2], axis1[2][2]], lw=1, color=:red)
        #plot!([axis2[1][1], axis2[2][1]], [axis2[1][2], axis2[2][2]], lw=1, color=:green)
    end

    display(current())
end


# Banana figures

visualize_velocity_space(target_list["Banana"],-3,3,-3,3,15,0.2)
savefig("velocity_space_full_banana.png")

visualize_velocity_space_at_points(target_list["Banana"],[[0.,1.],[-0.8,-0.],[0.8,0.],[-1.5,-1.5],[1.5,-1.5]],-3,3,-3,3,15,0.7)
savefig("velocity_space_partial_banana.png")

# Twin Peaks figures

visualize_velocity_space(target_list["Twin peaks"],-1,3,-2,2,11,0.2)
savefig("velocity_space_full_twin_peaks.png")

visualize_velocity_space_at_points(target_list["Twin peaks"],[[0.,0.],[1.,0.],[2.,0.],[0.,1.],[0.,-1.]],-1,3,-2,2,15,0.7)
savefig("velocity_space_partial_twin_peaks.png")
