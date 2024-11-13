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
    # Time to run the ODE for the data
    dataTime = 1000.0
    # Define the method specific settings and functions for MCMC
    prob = ODEProblem(Model.ode!, Model.ic_conv, (0.0, dataTime), Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e7)

    # Create the true data
    # True parameters
    pTrue = deepcopy(Model.params)
    pTrue = @set pTrue.g_Na_sf = 1.0
    pTrue = @set pTrue.g_K_sf = 1.0
    pTrue = @set pTrue.g_L_sf = 1.0

    # Run ODE to converged limit cycle
    prob_true = remake(prob, p=pTrue)::ODEProblem
    sol = DifferentialEquations.solve(prob_true, Tsit5(), maxiters=1e9)::ODESolution
    if Tools.auto_converge_check(prob_true, sol[end], pTrue)
        println("Data is appropriately converged")
    else
        println("Data was NOT generated from a converged limit cycle")
    end

    # Generate aligned data
    period = get_period(sol[end], prob_true)
    sol_pulse = Tools.aligned_sol(sol[end], prob_true, period)
    # Add noise and plot
    odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))
    plot(sol_pulse, title="True data"; label="Simulation")
    display(plot!(sol_pulse.t, odedata, label="Data"))

    # Save the data
    CSV.write("results/mcmc/data.csv", Tables.table([sol_pulse.t odedata]), writeheader=false)
end

function optimiseParameters()
    # Load data
    data = CSV.read("results/mcmc/data.csv", DataFrame)
    t = data[:, 1]
    data = data[:, 2]
    period = t[end]

    # Define optimisation variables and functions
    prob = ODEProblem(Model.ode!, Model.ic_conv, (0.0, 1000.0), Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e7)

    # Define model simulation function
    function model_simulator(p)
        prob = remake(prob, p=Tools.param_map(p))::ODEProblem
        # Converge
        sol = DifferentialEquations.solve(prob, Tsit5(), u0=Model.ic_conv, p=prob.p, tspan=(0.0, 1000.0), maxiters=1e9, save_everystep=false, save_start=false)
        sol_pulse = Tools.aligned_sol(sol(1000.0), prob, period)
        return sol_pulse.u
    end

    # Define the cost function
    function cost(p)
        sim = model_simulator(p)
        c = sum((data - sim).^2)
        return c
    end

    # Optimise
    p0 = [1.0, 1.0, 1.0]
    res = Optim.optimize(cost, p0, NelderMead(; initial_simplex=Optim.AffineSimplexer(b=0.0)), Optim.Options(show_trace=true))
    println(res.minimizer)
    return res
end

saveData()

optimiseParameters()
