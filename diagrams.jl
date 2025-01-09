using Plots, LaTeXStrings, Distributions
using CellMLToolkit, DifferentialEquations, ModelingToolkit, SymbolicIndexingInterface

include("./tools.jl")
using .Tools

include("./model.jl")
using .Model

plotParams = (linewidth=2., dpi=300)
l = @layout [a b]
# Panel A
initialCondition = 1.9
p1 = 0.15
p2 = 0.85
ylims = (0, 5)
lowerLine(x) = @. exp(-5*x+0.75)+0.5
upperLine(x) = @. exp(-6*x+3)+1
midLine(x) = @. exp(-5.5*x+2)+0.75
x = 0:0.01:1
y1 = lowerLine(x)
y2 = upperLine(x)
y3 = midLine(x)
plot(x, lowerLine, color=:black, label="", xlim=(0,1), ylim=ylims; plotParams...)
plot!(x, midLine, color=:black, label="", linestyle=:dot; plotParams...)
plot!(x, upperLine, color=:black, label="", xlabel="Parameter", ylabel="State", xticks=([p1, p2], [L"p_1", L"p_2"]), yticks=nothing; plotParams...)

function arrows(x; start=5)
    # Add NaNs into the vector x for where the arrows should be
    newx = convert(Vector, copy(x))
    for i in length(x)-start:-10:5
        insert!(newx, i, NaN)
        insert!(newx, i, x[i])
    end
    return newx
end

# Continuation curve
pointsOnStraight = 12
pointsOnCurve = 100
x1 = arrows(repeat([p1], pointsOnStraight))
y1 = arrows(LinRange(initialCondition, lowerLine(p1), pointsOnStraight))
x2 = arrows(LinRange(p1, p2, pointsOnCurve); start=10)
y2 = lowerLine(x2)
plot!(vcat(x1, x2), vcat(y1, y2), label="", color=:blue, arrow=true; plotParams...)

# ODE curve
pointsOnStraight = 12
x = arrows(repeat([p2], pointsOnStraight))
y = arrows(LinRange(initialCondition, upperLine(p2), pointsOnStraight))
plot!(x, y, label="", color=:red, arrow=true; plotParams...)

# Initial condition
plotA = hline!([initialCondition], label="", color=:pink, legend=false; plotParams...)

# Panel B
initialCondition = 0.53
p1 = 0.15
p2 = 0.85

centerx1 = 0.3
centerx2 = 0.7
centery1 = 0.25
centery2 = 0.75
curve1 = 8
curve2 = 20
xCurve1(y) = @. -curve1*(y-centery1)^2+centerx1
xCurve2(y) = @. curve2*(y-centery2)^2+centerx2

y = 0:0.01:centery1
plot(xCurve1(y), y, xlim=(0,1), ylim=(0,1), linestyle=:dot, color=:black, label=""; plotParams...)
y = centery1:0.01:1
plot(xCurve1(y), y, xlim=(0,1), ylim=(0,1), color=:black, label=""; plotParams...)
y = 0:0.01:centery2
plot!(xCurve2(y), y, color=:black, label="Stable LC"; plotParams...)
y = centery2:0.01:1
plot!(xCurve2(y), y, color=:black, label="Unstable LC", linestyle=:dot, xticks=([p1, p2], [L"p_1", L"p_2"]), yticks=nothing; plotParams...)
xlabel!("Parameter")

# Continuation curve
criticalY = centery1+sqrt((centerx1-p1)/curve1)
pointsOnStraight = 12
pointsOnCurve = 45
x1 = arrows(repeat([p1], pointsOnStraight))
y1 = arrows(LinRange(initialCondition, criticalY, pointsOnStraight))
y2 = arrows(LinRange(criticalY, centery1-sqrt(centerx1/curve1), pointsOnCurve); start=7)
x2 = xCurve1(y2)
x = vcat(x1, x2)
y = vcat(y1, y2)
plot!(x, y, label="Continuation", color=:blue, arrow=true; plotParams...)

# ODE curve
pointsOnStraight = 12
x = arrows(repeat([p2], pointsOnStraight))
y = arrows(LinRange(initialCondition, centery2-sqrt((p2-centerx2)/curve2), pointsOnStraight))
plot!(x, y, label="Standard", color=:red, arrow=true, legend=:bottomright; plotParams...)

# Initial Condition
plotB = hline!([initialCondition], label="IC", color=:pink; plotParams...)

plot(plotA, plotB, layout=l, size=(539,250), dpi=300, left_margin=4Plots.mm, 
title=["A" "B"], titlelocation=:left)
savefig("results/diagrams/possibleProblems.pdf")

ml = CellModel("ohara_rudy_cipa_v1_2017.cellml")
ic = [7.8e-5, -88.0]
prob = ODEProblem(ml.sys, [], (0,50000.0), [ml.sys.intracellular_ions₊ki=>140.0, ml.sys.intracellular_ions₊nai=>6, ml.sys.membrane₊v=>ic[2], ml.sys.intracellular_ions₊cai=>ic[1]], abstol=1e-10, reltol=1e-8)
sol = solve(prob, Tsit5(), saveat=1.0, maxiters=1e9)

plot(sol.t.-49000.0, sol[variable_index(ml.sys, ml.sys.membrane₊v),:], color=:black, legend=nothing; plotParams...)
plot!(sol.t, sol[variable_index(ml.sys, ml.sys.membrane₊v),:], color=:hotpink; plotParams...)
xlims!(0.0, 400.0)
xlabel!("Time (ms)")
plotA = ylabel!("Membrane Voltage (mV)")

my_cgrad = cgrad([:hotpink, :black])
plot(sol, idxs=(ml.sys.intracellular_ions₊cai,ml.sys.membrane₊v), legend=nothing; lc=my_cgrad, line_z=sol.t, plotParams...)
xlabel!("Intracellular Ca²⁺ (μM)")
xaxis!(xformatter=x->x*1e3)
plotB = scatter!([ic[1]], [ic[2]], color=:hotpink; plotParams...)

plot(plotA, plotB, layout=l, size=(539,250), dpi=300, bottom_margin=2Plots.mm,
title=["A" "B"], titlelocation=:left, link=:y)
savefig("results/diagrams/actionPotential.pdf")

# Limit cycles
plotTime = 1.0
# Start - Tracking
params = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0, conv_rate=2.5)
prob_de = ODEProblem(Model.ode!, Model.ic, (0.,1000.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, saveat=10.0)
sol = Tools.aligned_sol(sol[end], prob_de, plotTime)
plot(sol, label="Start - Tracking", idxs=(1); plotParams...)

# Start - Standard
params = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0, conv_rate=2.5)
prob_de = ODEProblem(Model.ode!, Model.ic, (0.,1000.0), params, reltol=1e-8, abstol=1e-10)
sol = Tools.aligned_sol(Model.ic, prob_de, plotTime)
plot!(sol, label="Start - Standard", idxs=(1); plotParams...)

# End - Small perturbation
params = (g_Na_sf=1.1, g_K_sf=1.0, g_L_sf=1.0, conv_rate=2.5)
prob_de = ODEProblem(Model.ode!, Model.ic, (0.,1000.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, saveat=10.0)
sol = Tools.aligned_sol(sol[end], prob_de, plotTime)
plot!(sol, label="End - Small Perturbation", idxs=(1); plotParams...)

# End - Large perturbation
params = (g_Na_sf=1.5, g_K_sf=1.2, g_L_sf=0.8, conv_rate=2.5)
prob_de = ODEProblem(Model.ode!, Model.ic, (0.,1000.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, saveat=10.0)
sol = Tools.aligned_sol(sol[end], prob_de, plotTime)
plot!(sol, label="End - Large Perturbation", idxs=(1), legend=:outerbottom, legend_columns=2; plotParams...)

xaxis!(xformatter = x -> x*1000)
xlabel!("Time (ms)")
ylabel!("Voltage (mV)")
title!("Variation in APs")
plot!(size=(539,300), dpi=300, rightmargin=2Plots.mm, bottommargin=-8Plots.mm)
savefig("results/diagrams/limitCycles.pdf")
