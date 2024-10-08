# Metropolis Hastings MCMC to be used for both ODE solver and continuation solver
# Also stores and passes the current limit cycle to the next iteration
using Distributions, LinearAlgebra
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

function mcmc(numSamples::Int64, solver::Function, μ₀::Vector{Float64}, prob::ODEProblem, data::Vector{Float64}, paramMap::Function, verbose::Int64=1)
    # verbose : int
    #     The verbosity level. 0 is silent, 1 is standard, 2 is debug
    # Set up
    chain = zeros(numSamples, length(μ₀))
    accepts = zeros(numSamples)
    x = copy(μ₀)
    σ = x[end]
    a = 1.0
    adaptionStart = ceil(numSamples*0.1) # Start adaptive covariance after 10% of samples
    Σ = Hermitian(diagm(μ₀/100))
    prob = remake(prob, p=paramMap(x, x), u0=Model.ic_conv)
    lc = solver(x, prob, Model.ic_conv, x, paramMap, verbose)
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
        xNew = q(x, a*Σ)
        σNew = xNew[end]
        # Solve with new parameters
        lcNew = solver(xNew, prob, lc, x, paramMap, verbose)
        if verbose > 1
            println("Proposed parameters: ", xNew[1:end-1])
            println("Proposed noise: ", σNew)
            println("Proposed limit cycle: ", lcNew)
        end
        # Calculate acceptance probability
        llNew = ll(lcNew, data, σNew, remake(prob, p=paramMap(xNew, x)))
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
            accepts[i] = 1
        else
            if verbose > 0
                println("- Proposal rejected")
            end
            accepts[i] = 0
        end
        if verbose > 0 && i > 100
            println("Local acceptance rate (%): ", sum(accepts[i-100:i]))
        end
        chain[i, :] = x
        if i == adaptionStart + 1 && verbose > 0
            println("Adaption started")
        end
        if i > adaptionStart
            s = i - adaptionStart
            γ = (s+1)^-0.6
            Σ = Hermitian((1-γ)*Σ + γ*(x - μ₀)*(x - μ₀)')
            μ₀ = (1-γ)*μ₀ + γ*x
            a *= exp(γ*(sum(accepts[Int(adaptionStart)+1:i])/s - 0.25))
            if verbose > 0
                println("Adaption acceptance rate: ", sum(accepts[Int(adaptionStart)+1:i])/s)
            end
            if verbose > 1
                println("Adaption step: ", s)
                println("γ: ", γ)
                println("Σ: ", Σ)
                println("μ₀: ", μ₀)
                println("a: ", a)
            end
        end
    end
    return chain
end

function q(x::Vector{Float64}, Σ::Hermitian{Float64})::Vector{Float64}
    # Perturb state x to get a new state xNew and return it
    d = MvNormal(x, Σ)
    return rand(d)
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

function odeSolverFull(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, verbose::Int64)::Vector{Float64}
    # Solve the ODE until convergence starting from the default initial conditions
    prob = remake(prob, p=paramMap(x, xlc))
    tmp = solve(prob, Tsit5(), maxiters=1e7, save_everystep = false; tspan=(0.0, 50000.0), p=paramMap(x, xlc)) #Should go to 50000 for full run
    return tmp[end]
end

function odeSolverCheap(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, verbose::Int64)::Vector{Float64}
    # Solve the ODE until convergence but starting from the previous limit cycle
    prob = remake(prob, p=paramMap(x, xlc))
    tmp = solve(prob, Tsit5(), maxiters=1e7, save_everystep = false; tspan=(0.0, 10000.0), p=paramMap(x, xlc), u0=lc)
    return tmp[end]
end

function contSolver(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, bp::BifurcationProblem, verbose::Int64)::Vector{Float64}
    # Perform continuation on the ODE to get the limit cycle
    # x: The parameters to find the limit cycle for
    # prob: The ODEProblem to solve during continuation
    # lc: The previous limit cycle
    # xlc: The parameters of the previous limit cycle
    # bp: The BifurcationProblem to solve during continuation
    # verbose: The verbosity level (0 silent, 1 standard, 2 debug)
    # Remake BP and prob
    bp = re_make(bp; u0=lc, params=paramMap(x, xlc))
    prob = remake(prob, u0=lc, p=paramMap(x, xlc))
    # Create a solution using the previous limit cycle
    sol = solve(prob, Tsit5(), maxiters=1e7; tspan=(0.0, 50.0))
    # Shooting method
    bpsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
    bp, prob, sol, period; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

    opts_br = ContinuationPar(p_min = 0.0, p_max = 1.0, max_steps = 150, tol_stability = 1e-8, ds=1.0, dsmax=1.0, 
    detect_bifurcation=0, detect_fold=false,)

    bothside = xlc != x
    local brpo_sh
    try
        brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
    verbosity = 0, bothside=bothside)
    catch e
        if verbose > 0
            println("Error: ", e)
            println("Falling back to ODE solver")
        end
        return odeSolverCheap(x, prob, lc, xlc, paramMap, verbose)
    end
    if brpo_sh.sol[end].p == 1.0
        return brpo_sh.sol[end].x[1:5]
    elseif brpo_sh.sol[1].p == 1.0
        return brpo_sh.sol[1].x[1:5]
    else
        if verbose > 0
            println("First point: ", brpo_sh.sol[1].p)
            println("Last point: ", brpo_sh.sol[end].p)
            println("No cont point had parameters wanted. Falling back to ODE solver")
        end
        return odeSolverCheap(x, prob, lc, xlc, paramMap, verbose)
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

function param_map(x::Vector{Float64})::NamedTuple
    p = Model.params
    p = @set p.gna = x[1]
    p = @set p.gk = x[2]
    p = @set p.gs = x[3]
    p = @set p.gl = x[4]
    return p
end

function param_map_cont(x::Vector{Float64}, xlc::Vector{Float64})::NamedTuple
    # x: The parameters to find the limit cycle for
    # xlc: The parameters of the previous limit cycle
    p = Model.params_cont
    p = @set p.gna = xlc[1]
    p = @set p.gk = xlc[2]
    p = @set p.gs = xlc[3]
    p = @set p.gl = xlc[4]
    p = @set p.gna_step = x[1] - xlc[1]
    p = @set p.gk_step = x[2] - xlc[2]
    p = @set p.gs_step = x[3] - xlc[3]
    p = @set p.gl_step = x[4] - xlc[4]
    return p
end

use_continuation = false
paramMap(x,y) = use_continuation ? param_map_cont(x,y) : param_map(x)
prob = ODEProblem(use_continuation ? Model.ode_cont! : Model.ode!, Model.ic, (0.0, 50000.0), 
    use_continuation ? Model.params_cont : Model.params, abstol=1e-10, reltol=1e-8)
# # Parameters we will try to fit to
p = use_continuation ? Model.params_cont : Model.params
p = @set p.gna = 110.0
p = @set p.gk = 11.0
p = @set p.gs = 12.0
p = @set p.gl = 0.25
prob_true = remake(prob, p=p)
sol = solve(prob_true, Tsit5(), maxiters=1e7)
display(plot(sol, idxs=Model.slow_idx, title="Check limit cycle is converged for true data"))

sol_pulse, period = aligned_sol(sol, prob_true,)
odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))
plot(sol_pulse, title="True data"; label="Simulation")
display(plot!(sol_pulse.t, odedata, label="Data"))

# Set up continuation solver
lens = (@lens _.step)
bp = BifurcationProblem(Model.ode_cont!, Model.ic_conv, Model.params_cont, lens)

println("Log likelihood of true parameters: ", ll(sol.u[end], odedata, 2.0, prob_true))
solver(v, w, x, y, z, verbose) = use_continuation ? contSolver(v, w, x, y, z, bp, verbose) : odeSolverCheap(v, w, x, y, z, verbose)
numSamples = 1000*5*10 # 1000 samples per parameter before adaption (10% of the samples)
chain = mcmc(numSamples, solver, [120.0, 13.0, 10.0, 0.3, 1.5], prob, odedata, paramMap, 1)
# TODO: How long should ode solver be run for? Check how long it takes to converge max step of perturbation kernal

# Remove burn in stage
burnIn = Int(round(numSamples*0.25))
posterior = chain[burnIn+1:end, :]

# Plot posterior
paramNames = ["gna" "gk" "gs" "gl" "σ"]
for i in axes(posterior, 2)
    histogram(posterior[:, i], normalize=:pdf)
    if i == size(posterior, 2)
        title!("Noise")
    else
        title!("Parameter "*paramNames[i])
    end
    display(ylabel!("P(x)"))
end

plot((chain'./[110.0, 11.0, 12.0, 0.25, 2.0])', label=paramNames, title="Parameter and noise convergence", xlabel="Iteration", ylabel="Parameter value (relative to true)")
