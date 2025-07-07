using DifferentialEquations, DiffEqCallbacks
using BenchmarkTools
using Random

include("cipa.jl")
using .Cipa

prob = Cipa.prob

const debug = true
const nParameters = 100 # How many parameter vectors to use for the benchmark

# Setup
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

# Parameters
Random.seed!(0)
params = [0.85 .+ rand(13) * 0.3 for _ in 1:nParameters]
@show params

# ODE Convergence - Standard
println("Standard Approach")
for i in eachindex(params)
    println("Standard for parameter vector $i")
    prob_de = prob
    param_map!(prob_de, params[i])
    sol = DifferentialEquations.solve(prob_de, Tsit5(); callback=cb, Cipa.solversettings(save=false, maxt=2e6)...)
    display(sol.t)
    @show sol.u[end]
end

flush(stdout)

println("Running standard approach benchmark")
Random.seed!(0)
b = @benchmarkable DifferentialEquations.solve(prob, $Tsit5(); callback=$cb, $Cipa.solversettings(save=false, maxt=2e6)...) setup = (param_map!(prob, 0.85 .+ rand(13)*0.3))
t = run(b, seconds=3600*15, samples=nParameters, evals=1)
if length(t.times) != nParameters
    @show t
    throw("Not all parameters were run. $(length(t.times)) != $nParameters")
end
BenchmarkTools.save("results/cipa/simTimings/standard.json", t)

# ODE Convergence - Tracking
println("Tracking Approach")
for i in eachindex(params)
    println("Tracking for parameter vector $i")
    prob_de = remake(prob, u0=ic_conv)
    param_map!(prob_de, params[i])
    sol = DifferentialEquations.solve(prob_de, Tsit5(); callback=cb, Cipa.solversettings(save=false, maxt=2e6)...)
    display(sol.t)
    @show sol.u[end]
end

flush(stdout)

println("Running tracking approach benchmark")
Random.seed!(0)
b = @benchmarkable DifferentialEquations.solve(prob, $Tsit5(); callback=$cb, $Cipa.solversettings(save=false, maxt=2e6)...) setup = (param_map!(prob, 0.85 .+ rand(13)*0.3))
t = run(b, seconds=3600*15, samples=nParameters, evals=1)
if length(t.times) != nParameters
    @show t
    throw("Not all parameters were run. $(length(t.times)) != $nParameters")
end
BenchmarkTools.save("results/cipa/simTimings/tracking.json", t)

for i in eachindex(params)
    println("Continuation Approach")
    println("Continuation for parameter vector $i")
    lc = Cipa.continuation(Cipa.ic_conv, ones(13),
    params[i], debug)
    @show lc
end

flush(stdout)

println("Running continuation approach benchmark")
Random.seed!(0)
b = @benchmarkable Cipa.continuation($Cipa.ic_conv, $ones(13),
    p, false) setup = (p = 0.85 .+ rand(13)*0.3)
t = run(b, seconds=3600*15, samples=nParameters, evals=1)
if length(t.times) != nParameters
    @show t
    throw("Not all parameters were run. $(length(t.times)) != $nParameters")
end
BenchmarkTools.save("results/cipa/simTimings/continuation.json", t)
