using Distributions, LinearAlgebra, Random
using BifurcationKit, DifferentialEquations, BenchmarkTools
using CSV, Tables, Plots
using Optim, DataFrames

include("./model.jl")
using .Model

include("./tools.jl")
using .Tools

# Set seed
Random.seed!(1)

"""
    get_period(lc::Vector{Float64}, prob::ODEProblem)::Number

Get the period of the limit cycle.

# Arguments
- `lc::Vector{Float64}`: The limit cycle.
- `prob::ODEProblem`: The ODEProblem for the model.

# Returns
- `period::Number`: The period of the limit cycle.
"""
function get_period(lc::Vector{Float64}, prob::ODEProblem)::Number
    # Long simulation
    sol = DifferentialEquations.solve(prob, Tsit5(), tspan=(0.0, 100.0), u0=lc, maxiters=1e9)::ODESolution
    # Get local maximums
    maxs = []
    for i in 2:length(sol.t)-1
        if sol[1, i] > sol[1, i+1] && sol[1, i] > sol[1, i-1]
            push!(maxs, i)
        end
    end
    # Get average period (time between maxs)
    pulse_widths = [sol.t[maxs[i]]-sol.t[maxs[i-1]] for i in 2:length(maxs)]
    period = mean(pulse_widths)
    return period
end

function saveData()
    # Define the method specific settings and functions for MCMC
    prob = ODEProblem(Model.ode!, Model.ic, (0.0, 1000.0), Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e7)

    # Create the true data
    # True parameters
    pTrue = Tools.param_map([1.0, 1.0, 1.0])
    prob = remake(prob, p=pTrue)::ODEProblem

    # Run ODE to converged limit cycle
    sol = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e9, save_everystep=false)::ODESolution
    if Tools.auto_converge_check(prob, sol[end], pTrue)
        println("Data is appropriately converged")
    else
        println("Data was NOT generated from a converged limit cycle")
    end

    # Generate aligned data
    period = floor(get_period(sol[end], prob)/0.001)*0.001
    sol_pulse = Tools.aligned_sol(sol[end], prob, period)
    # Add noise and plot
    odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))

    # Save the data
    CSV.write("results/mcmc/data.csv", Tables.table([sol_pulse.t odedata]), writeheader=false)

    return sol_pulse, odedata, period
end

function ℓ(data, sol)
    σ = 2.0
    n = length(data)
    return -n*log(2π)/2 - n*log(σ^2)/2 - 1/(2σ^2)*sum((data - sol.u).^2)
end

function plotData(sol, data, mle, period)
    plot(sol, title="True data"; label="True solution - ℓ: "*string(round(ℓ(data, sol),sigdigits=4)))
    prob = ODEProblem(Model.ode!, Model.ic, (0.0, 1000.0), abstol=1e-10, reltol=1e-8, maxiters=1e7)
    prob = remake(prob, p=Tools.param_map(mle))::ODEProblem
    solMLE = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e9)::ODESolution
    solMLE = Tools.aligned_sol(solMLE[end], prob, period)
    plot!(solMLE, label="MLE - ℓ: "*string(round(ℓ(data, solMLE), sigdigits=4)))
    plot!(sol_pulse.t, data, label="Data")
    savefig("results/mcmc/data.pdf")
    return solMLE
end

function optimiseParameters()
    # Load data
    data = CSV.read("results/mcmc/data.csv", DataFrame, header=false)
    t = data[:, 1]
    data = data[:, 2]
    period = t[end]

    # Define optimisation variables and functions
    prob = ODEProblem(Model.ode!, Model.ic, (0.0, 1000.0); p=Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e9)

    # Define model simulation function
    function model_simulator(p)
        prob = remake(prob, p=Tools.param_map(p))::ODEProblem
        # Converge
        sol = DifferentialEquations.solve(prob, Tsit5(), save_everystep=false, save_start=false)
        sol_pulse = Tools.aligned_sol(sol[end], prob, period)
        return sol_pulse
    end

    # Define the cost function
    function cost(p)
        sim = model_simulator(p)
        return -ℓ(data, sim)
    end

    # Optimise
    p0 = [1.0, 1.0, 1.0]
    res = Optim.optimize(cost, p0, NelderMead(; initial_simplex=Optim.AffineSimplexer(b=0.0)), Optim.Options(show_trace=true))
    println(res.minimizer)
    return res
end

sol_pulse, odedata, period = saveData()

res = optimiseParameters()

solMLE = plotData(sol_pulse, odedata, res.minimizer, period)
