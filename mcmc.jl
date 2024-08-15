using Parameters, Plots, ConstructionBase, Revise, Setfield
using BifurcationKit, DifferentialEquations, BenchmarkTools
using Turing, StatsPlots
using LinearAlgebra
include("./model.jl")
using .Model

prob = ODEProblem(Model.ode!, Model.ic, (0.0, 50000.0), Model.params, abstol=1e-10, reltol=1e-8)
# Parameters we will try to fit to
p = Model.params
p = @set p.gna = 100.0
prob_true = remake(prob, p=p)
sol = solve(prob_true, Tsit5(), maxiters=1e7)
plot(sol, idxs=Model.slow_idx, title="Check limit cycle is converged for true data")

function aligned_sol(sol, prob, period=0.0)
    mins = get_mins(sol)
    if period == 0.0
        period = get_period(mins)
        println("Period: ", period)
    end
    ic = sol[:, mins[end]]
    prob = remake(prob, u0=ic)
    return solve(prob, Tsit5(), saveat=0.1, save_idxs=Model.plot_idx, tspan=(0.0, period)), period
end

function get_mins(sol)
    mins = []
    for i in 2:length(sol.t)-1
        if sol[Model.plot_idx, i] < sol[Model.plot_idx, i+1] && sol[Model.plot_idx, i] < sol[Model.plot_idx, i-1]
            push!(mins, i)
        end
    end
    return mins
end

function get_period(mins)
    pulse_widths = [sol.t[mins[i]]-sol.t[mins[i-1]] for i in 2:length(mins)]
    period = mean(pulse_widths)
    return period
end

sol_pulse, period = aligned_sol(sol, prob_true,)
odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))
plot(sol_pulse, title="True data"; alpha=0.3)
plot!(sol_pulse.t, odedata)

@model function fit(data, prob1)
    # Prior distributions
    gna ~ Uniform(90.0, 150.0)
    σ ~ InverseGamma(2, 3)
    # Simulate model
    # Set parameters based on priors sampled
    p = Model.params
    p = @set p.gna = gna
    # Solve the ODE
    prob1 = remake(prob1, p=p)
    tmp = solve(prob1, Tsit5(), maxiters=1e7, save_everystep = false; p=p, tspan=(0.0, 10.0)) #Should go to 50000 for full run
    ic = tmp[end]
    tmp = solve(prob1, Tsit5(); p=p, u0=ic, tspan=(0.0, 50.0))
    predicted, = aligned_sol(tmp, prob1, period)

    # Observations
    data ~ MvNormal(predicted.u, σ^2 * I)

    return nothing
end

model = fit(odedata, prob)

chain = sample(model, NUTS(0.65), MCMCSerial(), 25, 1; progress=true) # For better convergence: 1000 rather than 25, 3 rather than 1
plot(chain)

# TODO: How can I store the previous limit cycle for the continuation method? - Can I do it in turing by storing them as parameters and defining transition
