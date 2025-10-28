using Plots, LaTeXStrings, Distributions
using CellMLToolkit, DifferentialEquations, ModelingToolkit, SymbolicIndexingInterface

include("./mcmcSetup.jl")

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

# Noble model concentration changes
prob_de = ODEProblem(Model.ode!, Model.ic_conv, (0.,10.0), params, reltol=1e-8, abstol=1e-10)
sol = Tools.aligned_sol(Model.ic_conv, prob_de, period; save_only_V=false)
# Shift back in time for plotting because starting at V=-20mV doesn't look nice
u0 = sol(0.4)
prob_de = remake(prob_de, u0=u0)
sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0.0, period), maxiters=1e9)
plot(sol, idxs=1, label="With Ions"; plotParams...)

# Do the same with the non-concentration model

function noble!(dz, z, p, t=0)
    @unpack g_Na_sf, g_K_sf, g_L_sf = p

    V, m, h, n = z

    alpha_m = 100 * (-V - 48) / (exp((-V - 48) / 15) - 1)
    beta_m = 120 * (V + 8) / (exp((V + 8) / 5) - 1)
    alpha_h = 170 * exp((-V - 90) / 20)
    beta_h = 1000 / (1 + exp((-V - 42) / 10))
    alpha_n = 0.1 * (-V - 50) / (exp((-V - 50) / 10) - 1)
    beta_n = 2 * exp((-V - 90) / 80)

    g_Na = g_Na_sf * m^3 * h * 400000
    g_K1 = g_K_sf * 1200 * exp((-V - 90) / 50) + 15 * exp((V + 90) / 60)
    g_K2 = g_K_sf * 1200 * n^4

    i_Leak = g_L_sf * 75 * (V + 60)
    i_Na = (g_Na + 140) * (V - 40)
    i_K = (g_K1 + g_K2) * (V + 100)

    dz[1] = -(i_Na + i_K + i_Leak) / 12
    dz[2] = alpha_m * (1 - m) - beta_m * m
    dz[3] = alpha_h * (1 - h) - beta_h * h
    dz[4] = alpha_n * (1 - n) - beta_n * n

    dz
end

z0 = [-87.0, 0.01, 0.8, 0.01]
prob_de = ODEProblem(noble!, z0, (0.0, 500.0), params, reltol=1e-8, abstol=1e-10)
sol = Tools.aligned_sol(z0, prob_de, period; save_only_V=false)
# Shift back in time for plotting because starting at V=-20mV doesn't look nice, manually aligned with other AP
u0 = sol(0.564)
prob_de = remake(prob_de, u0=u0)
sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0.0, period), maxiters=1e9)
p1 = plot!(sol, idxs=1, label="Without Ions", xlabel="Time (ms)",
xformatter=x -> x * 1000, ylabel = "Voltage (mV)"; plotParams...)

# Second plot for concentrations
prob_de = ODEProblem(Model.ode!, Model.ic_conv, (0., 10.0), params, reltol=1e-8, abstol=1e-10)
sol = Tools.aligned_sol(Model.ic_conv, prob_de, period; save_only_V=false)
# Shift back in time for plotting because starting at V=-20mV doesn't look nice
u0 = sol(0.4)
prob_de = remake(prob_de, u0=u0)
sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0.0, period), maxiters=1e9)
p2 = plot(sol, idxs=[5], label="Na", ylabel="Na Concentration", xlabel="Time (ms)",
xformatter=x -> x * 1000, legend=false; plotParams...)
p2t = twinx()
plot!(p2t, sol, idxs=[6], label=nothing, xlabel="", ylabel="K Concentration", color=:red, legend=false; plotParams...)
plot!(p2, [NaN], [NaN], label="K", color=:red; plotParams...)
plot!(p2, legend=:best)

# Plot of voltage before and after convergence
prob_de = ODEProblem(Model.ode!, Model.ic, (0., 10.0), params, reltol=1e-8, abstol=1e-10)
sol = Tools.aligned_sol(Model.ic, prob_de, period; save_only_V=false)
u0 = sol(0.5)
prob_de = remake(prob_de, u0=u0)
sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0.0, period), maxiters=1e9)
plot(sol, idxs=1, label="Unconverged"; plotParams...)
prob_de = ODEProblem(Model.ode!, Model.ic, (0., 200.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e9)
sol = Tools.aligned_sol(sol[end], prob_de, period; save_only_V=false)
u0 = sol(0.3475)
prob_de = remake(prob_de, u0=u0)
sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0.0, period), maxiters=1e9)
p3 = plot!(sol, idxs=1, label="Converged", xlabel="Time (ms)", xformatter=x -> x * 1000,
ylabel="Voltage (mV)"; plotParams...)

# Plot of concentrations during convergence
prob_de = ODEProblem(Model.ode!, Model.ic, (0., 100.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e9)
p4 = plot(sol, idxs=[5], label="Na", ylabel="Na Concentration", xlabel="Time (s)",
legend=false; plotParams...)
p4t = twinx()
plot!(p4t, sol, idxs=[6], label=nothing, ylabel="K Concentration", xlabel="",
color=:red, legend=false; plotParams...)
plot!(p4, [NaN], [NaN], label="K", color=:red; plotParams...)
plot!(p4, legend=:best)

l_vertical = @layout [a c; b d]
plot(p1, p2, p3, p4, layout=l_vertical, size=(700,600), title=["A" "B" "" "C" "D" ""],
titlelocation=:left, dpi=300)
savefig("results/diagrams/nobleConc.pdf")
