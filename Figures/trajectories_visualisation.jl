include("../main.jl")
include("../Tests/Twin peaks.jl");
target_list = Dict("Twin peaks" => target)
include("../Tests/Banana.jl")
target_list["Banana"] = target

using LinearAlgebra
using Plots
using Measures
using Profile

target = target_rosenbrock#target_list["Twin peaks"]

pdmp = PDMP(Lagrangian(0.1), target, false)
nums = LagrangianNumerics(position_method = VTPiecewiseConstant(0.01))

#pdmp = PDMP(BPS(), target, false)
#nums = BPSNumerics(position_method = VTAdaptivePiecewiseConstant(0.001,0.01,10000));

evo_data = initialize_evolution_data(pdmp)


state = initialize_binary_state!(pdmp, evo_data, initial_position = [0.,0.]);
acc, skeleton = new_point!(pdmp,reverse(pdmp),state,evo_data,nums,100.0,true)
print(length(skeleton))
print(acc)
using Plots

# Extract x and y coordinates
x = [p[1] for p in skeleton]
y = [p[2] for p in skeleton]

# Plot lines connecting consecutive points

xmin = -10
xmax = 10
ymin = -5
ymax = 30
heatmap_grid_size = 1000
x_heat = range(xmin, xmax; length=heatmap_grid_size)
y_heat = range(ymin, ymax; length=heatmap_grid_size)

# === Evaluate g over fine grid ===
g_values = [exp(target.log_density([x, y])) for y in y_heat, x in x_heat]  # note: row-major (y, x)

# === Plot ===
heatmap(
    x_heat, y_heat, g_values;
    c=:grays,
    aspect_ratio=1,
    xlabel="x", ylabel="y",
    legend=false,
    #title=plot_title,
)

plot!(x, y; lw=2, color=:blue, label="", aspect_ratio=:equal)

# Overlay the points
scatter!(x, y; color=:red, markersize=5)