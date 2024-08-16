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

function aligned_sol(sol, prob, period=0.0; save_only_V = true)
    mins = get_mins(sol)
    if period == 0.0
        period = get_period(mins)
        println("Period: ", period)
    end
    ic = sol[:, mins[end]]
    prob = remake(prob, u0=ic)
    if save_only_V
        return solve(prob, Tsit5(), saveat=0.1, save_idxs=Model.plot_idx, tspan=(0.0, period)), period
    else
        return solve(prob, Tsit5(), saveat=0.1, tspan=(0.0, period)), period
    end
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

@model function fit(data, prob1) # prob1 does not mutate
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
# Could define a callback in turing and store the current limit cycle globally?
@model function fit_cont(data, bp, sol_pulse, prob, opts_br)
    # Prior distributions
    gna ~ Uniform(90.0, 150.0)
    σ ~ InverseGamma(2, 3)
    # Simulate model
    # Set parameters based on priors sampled
    println("Starting markov chain sample")
    println("We want to try gna: ", gna)
    println("Default value of gna: ", bp.params.gna)
    println("We are currently working with the bounds: ", opts_br.p_min, " and ", opts_br.p_max)
    # Track the limit cycle
    if gna < bp.params.gna
        opts_br = setproperties(opts_br, p_min = gna, p_max = bp.params.gna) # Breaks here because the sampled gna is an autodiff object
    else
        opts_br = setproperties(opts_br, p_min = bp.params.gna, p_max = gna)
    end
    println("We have now updated the bounds to be: ", opts_br.p_min, " and ", opts_br.p_max)
    println("Current value of gna in prob: ", prob.p.gna)
    bpsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
    bp, prob, sol_pulse, period; alg = Tsit5(), abstol=1e-10, reltol=1e-8)
    
    brpo_sh = continuation(bpsh, cish, PALC(), opts_br) # TODO: Continuation might not reach bounds
    println("Continuation has finished")
    println(brpo_sh)
    println("Generating limit cycle from ODE")
    ic = brpo_sh.sol[2].x[1:5]
    p = Model.params
    p = @set p.gna = gna
    prob = remake(prob, u0=ic, p=p)
    println("Updated value of gna in prob: ", prob.p.gna)
    tmp = solve(prob, Tsit5(); tspan=(0.0, 50.0))
    predicted, = aligned_sol(tmp, prob, period)

    # Observations
    data ~ MvNormal(predicted.u, σ^2 * I)
    println("Sample complete")
    return nothing
end

prob = remake(prob, u0 = Model.ic_conv, tspan=(0.0, 50.0))
sol = solve(prob, Tsit5())
sol_pulse, = aligned_sol(sol, prob, period, save_only_V=false)

lens = Model.cont_params[1]
bp = BifurcationProblem(Model.ode!, Model.ic_conv, Model.params, lens;
	record_from_solution = (x, p) -> (V = x[plot_idx]),)

opts_br = ContinuationPar(p_min = 90.0, p_max = 150.0, max_steps = 50, tol_stability = 1e-8, ds=1.0, dsmax=60.0, 
    detect_bifurcation=0, detect_fold=false,)
model_cont = fit_cont(odedata, bp, sol_pulse, prob, opts_br)

chain = sample(model_cont, NUTS(0.65), MCMCSerial(), 25, 1; progress=true) # For better convergence: 1000 rather than 25, 3 rather than 1
plot(chain)
