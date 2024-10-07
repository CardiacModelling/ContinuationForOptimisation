# Metropolis Hastings MCMC to be used for both ODE solver and continuation solver
# Also stores and passes the current limit cycle to the next iteration
using Distributions
using Parameters, Plots, ConstructionBase, Revise, Setfield
using BifurcationKit, DifferentialEquations, BenchmarkTools

include("./model.jl")
using .Model

# State x: [conductance parameters, noise σ]
# State-ish lc: [initial conditions for limit cycle]
# Solver is a function that takes in a state x and returns a point on the converged limit cycle
# Init is the initial state x - [conductance parameters, noise σ]
# prob is an ODEProblem for computing the full limit cycle and likelihood
# data is the data to compare the output of solver with

function mcmc(numSamples::Int64, solver::Function, initP::Vector{Float64}, prob::ODEProblem, data::Vector{Float64}, verbose::Int64=1)
    # verbose : int
    #     The verbosity level. 0 is silent, 1 is standard, 2 is debug
    # Set up
    chain = zeros(numSamples, length(initP))
    x = copy(initP)
    σ = x[end]
    prob = remake(prob, p=paramMap(x), u0=Model.ic_conv)
    lc = solver(x, prob, Model.ic_conv, x, verbose)
    llOld = ll(lc, data, σ, prob)
    if verbose > 0
        println("========        Starting MCMC        ========")
        println("Initial parameters: ", x[1:end-1])
        println("Initial noise: ", σ)
        println("Initial limit cycle: ", lc)
        println("Number of samples: ", numSamples)
        println("=============================================")
    end
    # Iterate through MCMC steps
    for i in 1:numSamples
        if verbose > 0
        println("========         Iteration ", i, "         ========")
        println("Current parameters: ", x[1:end-1])
        println("Current noise: ", σ)
        println("Current limit cycle: ", lc)
        end
        # Sample from proposal
        xNew = q(x, initP)
        σNew = xNew[end]
        # Solve with new parameters
        lcNew = solver(xNew, prob, lc, x, verbose)
        if verbose > 1
            println("Proposed parameters: ", xNew[1:end-1])
            println("Proposed noise: ", σNew)
            println("Proposed limit cycle: ", lcNew)
        end
        # Calculate acceptance probability
        llNew = ll(lcNew, data, σNew, remake(prob, p=paramMap(xNew)))
        α = min(1, exp(π(xNew) + llNew - π(x) - llOld)) # Assuming proposal kernal q(xNew|x) is symmetric
        if verbose > 1
            println("Acceptance probability: ", α)
            println("Current log likelihood: ", llOld)
            println("Proposed log likelihood: ", llNew)
        end
        # Accept or reject proposal - Metropolis-Hastings
        if rand() < α
            if verbose > 0
                println("+ Proposal accepted")
            end
            x = xNew
            σ = σNew
            lc = lcNew
            llOld = llNew
        else
            if verbose > 0
                println("- Proposal rejected")
            end
        end
        chain[i, :] = x
    end
    return chain
end

function q(x::Vector{Float64}, initP::Vector{Float64})::Vector{Float64}
    # Perturb state x to get a new state xNew and return it
    x += initP .* (randn(size(x)).-0.5) ./ 100.0
    return x
end

function π(x::Vector{Float64})::Float64
    # Calculate the prior logpdf of x
    # Uniform priors for conductance parameters - can ignore normalisation constant
    # Inverse Gamma prior for noise σ
    ig = InverseGamma(2, 3)
    return logpdf(ig, x[end])
end

function ll(limitCycle::Vector{Float64}, data::Vector{Float64}, σ::Float64, prob::ODEProblem)::Float64
    # Calculate the log-likelihood of the limit cycle compared with the data, and σ
    sol = solve(prob, Tsit5(), dtmax=0.001; tspan=(0.0, 100.0), u0=limitCycle)
    sol, = aligned_sol(sol, prob, period)
    # Calculate the likelihood of the data given the limit cycle
    n = Normal(0, σ)
    return loglikelihood(n, data - sol.u)
end

function aligned_sol(sol, prob::ODEProblem, period::Float64 = 0.0; save_only_V::Bool = true)
    # Align the limit cycle in the solution to start at the min of V
    maxs = get_maxs(sol)
    if period == 0.0
        period = get_period(maxs)
        println("Period: ", period)
    end
    ic = sol[:, maxs[end]]
    if save_only_V
        return solve(prob, Tsit5(), saveat=0.1, save_idxs=Model.plot_idx, tspan=(0.0, period), u0=ic), period
    else
        return solve(prob, Tsit5(), saveat=0.1, tspan=(0.0, period), u0=ic), period
    end
end

function odeSolverFull(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, verbose::Int64)::Vector{Float64}
    # Solve the ODE until convergence starting from the default initial conditions
    prob = remake(prob, p=paramMap(x))
    tmp = solve(prob, Tsit5(), maxiters=1e7, save_everystep = false; tspan=(0.0, 50000.0), p=paramMap(x)) #Should go to 50000 for full run
    return tmp[end]
end

function odeSolverCheap(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, verbose::Int64)::Vector{Float64}
    # Solve the ODE until convergence but starting from the previous limit cycle
    prob = remake(prob, p=paramMap(x))
    tmp = solve(prob, Tsit5(), maxiters=1e7, save_everystep = false; tspan=(0.0, 10000.0), p=paramMap(x), u0=lc)
    return tmp[end]
end

function contSolver(x::Vector{Float64}, prob::ODEProblem, xlc::Vector{Float64}, bp::BifurcationProblem, verbose::Int64)::Vector{Float64}
    # Perform continuation on the ODE to get the limit cycle
    # x: The parameters to find the limit cycle for
    # prob: The ODEProblem to solve during continuation
    # lc: The previous limit cycle
    # xlc: The parameters of the previous limit cycle
    # bp: The BifurcationProblem to solve during continuation
    # verbose: The verbosity level (0 silent, 1 standard, 2 debug)
    # Remake BP and prob
    bp = re_make(bp; u0=lc, params=paramMap(xlc))
    prob = remake(prob, u0=lc, p=paramMap(xlc))
    # Create a solution using the previous limit cycle
    sol = solve(prob, Tsit5(), maxiters=1e7; tspan=(0.0, 50.0))
    # Shooting method
    bpsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
    bp, prob, sol, period; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

    opts_br = ContinuationPar(p_min = min(x[1], xlc[1]), p_max = max(x[1], xlc[1]), max_steps = 150, tol_stability = 1e-8, ds=0.1*x[1], dsmax=0.1*x[1], 
    detect_bifurcation=0, detect_fold=false,)

    bothside = xlc[1] != x[1]
    local brpo_sh
    try
        brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
    verbosity = 0, bothside=bothside)
    catch e
        if verbose > 0
            println("Error: ", e)
            println("Falling back to ODE solver")
        end
        return odeSolverCheap(x, prob, lc, verbose)
    end
    if brpo_sh.sol[end].p == x[1]
        return brpo_sh.sol[end].x[1:5]
    elseif brpo_sh.sol[1].p == x[1]
        return brpo_sh.sol[1].x[1:5]
    else
        if verbose > 0
            println("First point: ", brpo_sh.sol[1].p)
            println("Last point: ", brpo_sh.sol[end].p)
            println("Parameters wanted: ", x[1])
            println("No cont point had parameters wanted. Falling back to ODE solver")
        end
        return odeSolverCheap(x, prob, lc, verbose)
    end
end

function get_maxs(sol)::Vector{Int64}
    maxs = []
    for i in 2:length(sol.t)-1
        if sol[Model.plot_idx, i] > sol[Model.plot_idx, i+1] && sol[Model.plot_idx, i] > sol[Model.plot_idx, i-1]
            push!(maxs, i)
        end
    end
    return maxs
end

function get_period(maxs::Vector{Int64})::Float64
    pulse_widths = [sol.t[maxs[i]]-sol.t[maxs[i-1]] for i in 2:length(maxs)]
    period = mean(pulse_widths)
    return period
end

function paramMap(x::Vector{Float64})::NamedTuple
    p = Model.params
    p = @set p.gna = x[1]
    return p
end

prob = ODEProblem(Model.ode!, Model.ic, (0.0, 50000.0), Model.params, abstol=1e-10, reltol=1e-8)
# # Parameters we will try to fit to
p = Model.params
p = @set p.gna = 110.0
prob_true = remake(prob, p=p)
sol = solve(prob_true, Tsit5(), maxiters=1e7)
display(plot(sol, idxs=Model.slow_idx, title="Check limit cycle is converged for true data"))

sol_pulse, period = aligned_sol(sol, prob_true,)
odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))
plot(sol_pulse, title="True data"; label="Simulation")
display(plot!(sol_pulse.t, odedata, label="Data"))

# Set up continuation solver
lens = Model.cont_params[1]
pVal = Setfield.get(Model.params, lens)
bp = BifurcationProblem(Model.ode!, Model.ic_conv, Model.params, lens)

println("Log likelihood of true parameters: ", ll(sol.u[end], odedata, 2, prob_true))
solver(x, y, z, verbose) = contSolver(x, y, z, bp, verbose)

@profview chain = mcmc(20, solver, [120.0, 1.5], prob, odedata, 2)
# TODO: How long should ode solver be run for? Check how long it takes to converge max step of perturbation kernal
# TODO: Long runs of continuation
# TODO: optimise contuation solver
# TODO: tidy code
# TODO: continuation seems to only operate between 115 and 120 - maybe, cant seem to reproduce, seems inconsistent
# TODO: Add other parameters
# TODO: Remove noise parameter?
# TODO: Track accepts and rejects over time

# Remove burn in stage
burnIn = 0
posterior = chain[burnIn+1:end, :]

# Plot posterior
for i in axes(posterior, 2)
    histogram(posterior[:, i], normalize=:pdf)
    if i == size(posterior, 2)
        title!("Noise")
    else
        title!("Parameter "*string(i))
    end
    display(ylabel!("P(x)"))
end

plot((chain'./[110.0, 2.0])', label=["gna" "σ"], title="Parameter and noise convergence", xlabel="Iteration", ylabel="Parameter value (relative to true)")
