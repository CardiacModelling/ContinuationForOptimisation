using Plots, LaTeXStrings, Distributions
using CellMLToolkit, DifferentialEquations, ModelingToolkit, SymbolicIndexingInterface

plotParams = (linewidth=2., dpi=300)
l = @layout [a b]
# Panel A
initialCondition = 1.9
p1 = 0.15
p2 = 0.85
ylims = (0, 5)
lowerLine(x) = @. exp(-5*x+0.75)+0.5
upperLine(x) = @. exp(-6*x+3)+1
x = 0:0.01:1
y1 = lowerLine(x)
y2 = upperLine(x)
plot(x, lowerLine, color=:black, label="", xlim=(0,1), ylim=ylims; plotParams...)
plot!(x, upperLine, color=:black, label="", xlabel="Parameter", ylabel="State", xticks=([p1, p2], [L"p_1", L"p_2"]), yticks=nothing; plotParams...)

# Continuation curve
plot!([p1, p1], [lowerLine(p1), initialCondition], label="", color=:blue; plotParams...)
plot!(p1:0.01:p2, lowerLine, color=:blue, label="Continuation"; plotParams...)

# ODE curve
plot!([p2, p2], [upperLine(p2), initialCondition], label="", color=:red; plotParams...)

# Initial condition
plotA = hline!([initialCondition], label="IC", color=:pink, legend=false; plotParams...)

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
plot!([p1, p1], [initialCondition, criticalY], label="", color=:blue; plotParams...)
y = criticalY:-0.01:centery1
plot!(xCurve1(y), y, color=:blue, label="Continuation"; plotParams...)
y = centery1:-0.01:0
plot!(xCurve1(y), y, color=:blue, linestyle=:dot, label=""; plotParams...)

# ODE curve
plot!([p2, p2], [initialCondition, centery2-sqrt((p2-centerx2)/curve2)], label="Full", color=:red, legend=:bottomright; plotParams...)

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
