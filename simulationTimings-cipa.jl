using Plots, BenchmarkPlots, StatsPlots
using DifferentialEquations, DiffEqCallbacks
using BenchmarkTools

include("cipa.jl")
using .Cipa

prob = Cipa.prob

const debug = true

# Define BenchmarkGroup
bg = BenchmarkGroup()
bg["Small"] = BenchmarkGroup()
bg["Large"] = BenchmarkGroup()
bg["Small"]["ODE"] = BenchmarkGroup()
bg["Small"]["Cont"] = BenchmarkGroup()
bg["Large"]["ODE"] = BenchmarkGroup()
bg["Large"]["Cont"] = BenchmarkGroup()

# Plot parameters
plot_params = (linewidth=2., dpi=300, size=(450, 300), legend=false)

# ODE Convergence - Standard
#params = [pSmall, pLarge]
params = [[1.05, 1.0, 1.025, 0.975, 1.05, 1.0, 1.0, 1.05, 1.05],
        [1.25, 1.0, 0.8, 1.25, 0.9, 0.75, 0.8, 0.9, 0.75]]
println("Standard Approach")
for i in eachindex(params)
    prob_de = prob
    STATE::Vector{Float64} = zeros(size(ic))
    function affect!(integrator)
        error = sum(abs.(STATE .- integrator.u))
        debug && println(error)
        if error < 1e-6
            terminate!(integrator)
        end
        STATE .= integrator.u
    end
    cb = PeriodicCallback(affect!, Cipa.pulse_period, save_positions=(false, false))
    param_map!(prob_de, params[i])
    sol = DifferentialEquations.solve(prob_de, Tsit5(); callback=cb, Cipa.solversettings(save=false, maxt=2e6)...)
    if i == 1
        println("Simulation time to convergence for small perturbation")
    else
        println("Simulation time to convergence for large perturbation")
    end
    display(sol.t)
    @show sol.u[end]
    b = @benchmarkable DifferentialEquations.solve($prob_de, $Tsit5(); callback=$cb, $Cipa.solversettings(save=false, maxt=2e6)...)
    bg[i == 1 ? "Small" : "Large"]["ODE"]["ODE - Standard"] = b
end

# ODE Convergence - Tracking
println("Tracking Approach")
for i in eachindex(params)
    condition(_, t, _) = (t + Cipa.pulse_period / 2) % Cipa.pulse_period - Cipa.pulse_period / 2
    STATE::Vector{Float64} = zeros(size(ic))
    function affect!(integrator)
        error = sum(abs.(STATE .- integrator.u))
        debug && println(error)
        if error < 1e-6
            terminate!(integrator)
        end
        STATE .= integrator.u
    end
    cb = PeriodicCallback(affect!, Cipa.pulse_period, save_positions=(false, false))
    prob_de = remake(prob, u0=ic_conv)
    param_map!(prob_de, params[i])
    sol = DifferentialEquations.solve(prob_de, Tsit5(); callback=cb, Cipa.solversettings(save=false, maxt=2e6)...)
    if i == 1
        println("Simulation time to convergence for small perturbation")
    else
        println("Simulation time to convergence for large perturbation")
    end
    display(sol.t)
    @show sol.u[end]
    b = @benchmarkable DifferentialEquations.solve($prob_de, $Tsit5(); callback=$cb, $Cipa.solversettings(save=false, maxt=2e6)...)
    bg[i == 1 ? "Small" : "Large"]["ODE"]["ODE - Tracking"] = b
end

for i in eachindex(params)
    println("Continuation Approach")
    if i == 1
        println("Continuation for small perturbation")
    else
        println("Continuation for large perturbation")
    end
    lc = Cipa.continuation(Cipa.ic_conv, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    params[i], debug)
    @show lc
    b = @benchmarkable Cipa.continuation($Cipa.ic_conv, $[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        $params[$i], $debug)
    bg[i == 1 ? "Small" : "Large"]["Cont"]["Continuator"] = b
end

println("Reached the end of the script. Just running benchmark now.")
t = run(bg, seconds=100)

BenchmarkTools.save("results/cipa/simTimings/data.json", t)
