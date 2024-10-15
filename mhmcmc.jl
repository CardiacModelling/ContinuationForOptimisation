# Metropolis Hastings MCMC to be used for both ODE solver and continuation solver
# Also stores and passes the current limit cycle to the next iteration
using Distributions, LinearAlgebra
using Parameters, Plots, ConstructionBase, Revise, Setfield
using BifurcationKit, DifferentialEquations, BenchmarkTools

include("./model.jl")
using .Model

include("./tools.jl")
using .Tools

"""
    converge(ic, solver, check, verbose=1)

Converge to a limit cycle using the `solver` function and verify it using `check` functions.

Returns `nothing` if the limit cycle was not found.

# Arguments
- `ic::Vector{Number}`: The initial conditions to start from.
- `solver::Function`: The function to solve the ODE.
- `check::Function`: The function to check if the limit cycle has converged.
- `verbose::Integer=1`: The verbosity level.

# Returns
- `lc::Vector{Number}`: The converged limit cycle
"""
function converge(ic::Vector{Number}, solver::Function, check::Function, verbose::Integer=1)::Union{Vector{Number}, Nothing}
    lc = copy(ic)
    for i in 1:5
        lc = solver(lc)
        if check(lc)
            if verbose > 1 && i > 1
                println("Required ", i, " iterations to converge")
            end
            return lc
        end
    end
    if verbose > 0
        println("Failed to converge to the limit cycle")
    end
    return nothing
end

"""
    mcmc(numSamples::Integer, solver::Function, μ₀::Vector{Number}, prob::ODEProblem, data::Vector{Number}, paramMap::Function, verbose::Integer=1)

Run an adaptive Metropolis-Hastings MCMC to find the posterior distribution of the parameters.

# Arguments
- `numSamples::Integer`: The number of samples to take.
- `solver::Function`: The function to compute the limit cycle.
- `μ₀::Vector{Number}`: The initial parameters to start from.
- `prob::ODEProblem`: The ODEProblem for the model.
- `data::Vector{Number}`: The data to compare the limit cycle to.
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose::Integer=1`: The verbosity level (0=None, 1=Minimal, 2=Standard, 3=Debug).

# Returns
- `chain::Matrix{Number}`: The chain of parameters.
- `accepts::Vector{Number}`: The acceptance rate of the proposals.
"""
function mcmc(numSamples::Integer, solver::Function, μ₀::Vector{Number}, prob::ODEProblem, data::Vector{Number}, paramMap::Function, verbose::Integer=1)
    # verbose : int
    #     The verbosity level. 0 is silent, 1 is minimal, 2 is standard, 3 is debug
    # Set up
    chain = zeros(numSamples, length(μ₀))
    accepts = zeros(numSamples)
    x = copy(μ₀)
    σ = x[end]
    a = 1.0
    adaptionStart = ceil(numSamples*0.1) # Start adaptive covariance after 10% of samples
    Σ = Hermitian(diagm(μ₀/10000))
    prob = remake(prob, p=paramMap(x, x), u0=Model.ic_conv)::ODEProblem
    lc = converge(Model.ic_conv, (ic) -> solver(x, prob, ic, x, paramMap, verbose), (ic) -> Tools.auto_converge_check(prob, ic, paramMap(x, x)), verbose)
    if lc === nothing
        lc = Model.ic_conv
        llOld = -Inf
    else
        llOld = ll(lc, data, σ, prob)
    end

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
        lcNew = converge(lc, (ic) -> solver(xNew, prob, ic, x, paramMap, verbose), (ic) -> Tools.auto_converge_check(prob, ic, paramMap(xNew, xNew)), verbose)
        if lcNew === nothing
            lcNew = lc
            llNew = -Inf
        else
            llNew = ll(lcNew, data, σNew, remake(prob, p=paramMap(xNew, x))::ODEProblem, verbose)
            if verbose > 2
                sol, = aligned_sol(lc, remake(prob, p=paramMap(x, x)), period)
                display(plot!(sol, label="Old"))
            end
        end
        if verbose > 1
            println("Proposed parameters: ", xNew[1:end-1])
            println("Proposed noise: ", σNew)
            println("Proposed limit cycle: ", lcNew)
        end
        # Calculate acceptance probability
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
                println("Current adaption acceptance rate: ", sum(accepts[Int(adaptionStart)+1:i])/s)
            end
            if verbose > 1
                println("Adaption step: ", s)
            end
            if verbose > 2
                println("γ: ", γ)
                println("Σ: ", Σ)
                println("μ₀: ", μ₀)
                println("a: ", a)
            end
        end
    end
    return chain, accepts
end

"""
    q(x::Vector{Number}, Σ::Hermitian{Number})::Vector{Number}

Perturb state `x` to get a new state `xNew` and return it.

# Arguments
- `x::Vector{Number}`: The current state.
- `Σ::Hermitian{Number}`: The covariance matrix.

# Returns
- `xNew::Vector{Number}`: The new state.
"""
function q(x::Vector{Number}, Σ::Hermitian{Number})::Vector{Number}
    # Perturb state x to get a new state xNew and return it
    d = MvNormal(x, Σ)
    return rand(d)
end

"""
    π(x::Vector{Number})::Number

Calculate the prior log probability density function of `x`.

Uniform priors for conductance parameters - can ignore normalisation constant.
Inverse Gamma prior for noise `σ`.

# Arguments
- `x::Vector{Number}`: The conductance parameters (uniform prior so ignored), and the noise parameter σ (`InverseGamma(2,3)` distribution).

# Returns
- `π::Number`: The prior logpdf.
"""
function π(x::Vector{Number})::Number
    ig = InverseGamma(2, 3)
    return logpdf(ig, x[end])
end

"""
    ll(limitCycle::Vector{Number}, data::Vector{Number}, σ::Number, prob::ODEProblem, verbose = 1)::Number

Calculate the log-likelihood of the limit cycle compared with the data, and σ.

# Arguments
- `limitCycle::Vector{Number}`: A point on the limit cycle to compare with the data.
- `data::Vector{Number}`: The data to compare with the limit cycle.
- `σ::Number`: The estimated noise standard deviation.
- `prob::ODEProblem`: The ODEProblem for the model.
- `verbose::Integer=1`: The verbosity level.

# Returns
- `ll::Number`: The log-likelihood.
"""
function ll(limitCycle::Vector{Number}, data::Vector{Number}, σ::Number, prob::ODEProblem, verbose = 1)::Number
    # Calculate the log-likelihood of the limit cycle compared with the data, and σ
    sol, = aligned_sol(limitCycle, prob, period)
    # Calculate the likelihood of the data given the limit cycle
    n = Normal(0, σ)
    if verbose > 2
        plot(sol, label="Proposed")
        plot!(sol.t, data, label="Data")
    end
    return loglikelihood(n, data - sol.u)
end

"""
    aligned_sol(lc::Vector{Number}, prob::ODEProblem, period::Number = 0.0; save_only_V::Bool = true)

Align the limit cycle in the solution to start at the max of V and fixes the timesteps for recording the data.

If the period is not specified, it will be calculated.

# Arguments
- `lc::Vector{Number}`: The limit cycle to align.
- `prob::ODEProblem`: The ODEProblem for the model.
- `period::Number=0.0`: The period of the data.
- `save_only_V::Bool=true`: Save only the V (voltage) variable.

# Returns
- `sol::ODESolution`: The aligned solution.
- `period::Number`: The period of the limit cycle.
"""
function aligned_sol(lc::Vector{Number}, prob::ODEProblem, period::Number = 0.0; save_only_V::Bool = true)
    # Align the limit cycle in the solution to start at the max of V
    if period == 0.0
        period = get_period(lc, prob)
        println("Period: ", period)
    end
    # Simulation of length 2*period
    sol = solve(prob, Tsit5(); tspan=(0.0, period*2.0), u0=lc, save_idxs=Model.plot_idx, saveat=0.01, dense=false)::ODESolution
    t = sol.t[argmax(sol.u)]
    sol = solve(prob, Tsit5(); tspan = (0.0,t), u0=sol.prob.u0, save_everystep=false, save_start=false)
    if save_only_V
        return solve(prob, Tsit5(), saveat=0.1, save_idxs=Model.plot_idx, tspan=(0.0, period), u0=sol.u[end])::ODESolution, period
    else
        return solve(prob, Tsit5(), saveat=0.1, tspan=(0.0, period), u0=sol.u[end])::ODESolution, period
    end
end

"""
    odeSolverFull(x::Vector{Number}, prob::ODEProblem, lc::Vector{Number}, xlc::Vector{Number}, paramMap::Function, verbose::Integer)::Vector{Number}

Solve the ODE until convergence starting from the default initial conditions.

# Arguments
- `x::Vector{Number}`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve.
- `_::Vector{Number}`: The previous limit cycle (unused).
- `_::Vector{Number}`: The parameters of the previous limit cycle (unused).
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc::Vector{Number}`: The converged limit cycle.
"""
function odeSolverFull(x::Vector{Number}, prob::ODEProblem, _::Vector{Number}, xlc::Vector{Number}, paramMap::Function, verbose=1::Integer)::Vector{Number}
    prob = remake(prob, p=paramMap(x, xlc))::ODEProblem
    tmp = solve(prob, Tsit5(), save_everystep = false; tspan=(0.0, 50000.0), p=paramMap(x, x), save_start=false)::ODESolution
    return tmp[end]
end

"""
    odeSolverCheap(x::Vector{Number}, prob::ODEProblem, lc::Vector{Number}, xlc::Vector{Number}, paramMap::Function, verbose::Integer)::Vector{Number}

Solve the ODE until convergence but starting from the previous limit cycle.

# Arguments
- `x::Vector{Number}`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve.
- `lc::Vector{Number}`: The previous limit cycle.
- `_::Vector{Number}`: The parameters of the previous limit cycle (unused).
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc::Vector{Number}`: The converged limit cycle.
"""
function odeSolverCheap(x::Vector{Number}, prob::ODEProblem, lc::Vector{Number}, _::Vector{Number}, paramMap::Function, verbose::Integer)::Vector{Number}
    # Solve the ODE until convergence but starting from the previous limit cycle
    tmp = solve(prob, Tsit5(), save_everystep = false; tspan=(0.0, 10000.0), p=paramMap(x, x), u0=lc, save_start=false)::ODESolution
    return tmp[end]
end

"""
    contSolver(x::Vector{Number}, prob::ODEProblem, lc::Vector{Number}, xlc::Vector{Number}, paramMap::Function, bp::BifurcationProblem, verbose::Integer)::Vector{Number}

Perform continuation on the ODE to get the limit cycle.

# Arguments
- `x::Vector{Number}`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve during continuation.
- `lc::Vector{Number}`: The previous limit cycle.
- `xlc::Vector{Number}`: The parameters of the previous limit cycle.
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `bp::BifurcationProblem`: The BifurcationProblem to solve during continuation.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc::Vector{Number}`: The converged limit cycle.
"""
function contSolver(x::Vector{Number}, prob::ODEProblem, lc::Vector{Number}, xlc::Vector{Number}, paramMap::Function, bp::BifurcationProblem, verbose::Integer)::Vector{Number}
    # Remake BP and prob
    bp = re_make(bp; u0=lc, params=paramMap(x, xlc))::BifurcationProblem
    prob = remake(prob, u0=lc, p=paramMap(x, xlc))::ODEProblem
    # Create a solution using the previous limit cycle
    sol = solve(prob, Tsit5(), tspan=(0.0, 50.0))::ODESolution
    # Shooting method
    bpsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
    bp, prob, sol, period; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

    bothside = xlc != x
    local brpo_sh::Union{Nothing, ContResult}
    try
        brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
            verbosity = 0, bothside=bothside)
    catch e
        if verbose > 0
            println("Continuation failed: ", e)
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

"""
    get_period(lc::Vector{Number}, prob::ODEProblem)::Number

Get the period of the limit cycle.

# Arguments
- `lc::Vector{Number}`: The limit cycle.
- `prob::ODEProblem`: The ODEProblem for the model.

# Returns
- `period::Number`: The period of the limit cycle.
"""
function get_period(lc::Vector{Number}, prob::ODEProblem)::Number
    # Long simulation
    sol = solve(prob, Tsit5(), tspan=(0.0, 50000.0), u0=lc)::ODESolution
    # Get local maximums
    maxs = []
    for i in 2:length(sol.t)-1
        if sol[1, i] > sol[1, i+1] && sol[1, i] > sol[1, i-1]
            push!(maxs, i)
        end
    end
    # Get average period
    pulse_widths = [sol.t[maxs[i]]-sol.t[maxs[i-1]] for i in 2:length(maxs)]
    period = mean(pulse_widths)
    return period
end

"""
    param_map(x::Vector{Number})::NamedTuple{(:gna, :gk, :gs, :gl), Tuple{Number, Number, Number, Number}}

Map the parameters from a `Vector` to a `NamedTuple`.

# Arguments
- `x::Vector{Number}`: The parameters in a `Vector` ordered as gna, gk, gs, gl. The noise parameter can be optionally included at the end.

# Returns
- `par::NamedTuple`: The parameters as a `NamedTuple`.
"""
function param_map(x::Vector{Number})::NamedTuple{(:gna, :gk, :gs, :gl), Tuple{Number, Number, Number, Number}}
    par = p
    par = @set par.gna = x[1]
    par = @set par.gk = x[2]
    par = @set par.gs = x[3]
    par = @set par.gl = x[4]
    return par
end

"""
    param_map(x::Vector{Number}, xlc::Vector{Number})::NamedTuple{(:gna, :gk, :gs, :gl, :gna_step, :gk_step, :gs_step, :gl_step, :step), Tuple{Number, Number, Number, Number, Number, Number, Number, Number, Number}}

Map the parameters from a `Vector` to a `NamedTuple`.

Specific to the continuation solver.

# Arguments
- `x::Vector{Number}`: The parameters to find the limit cycle for.
- `xlc::Vector{Number}`: The parameters of the previous limit cycle.

# Returns
- `par::NamedTuple`: The parameters as a `NamedTuple`.
"""
function param_map(x::Vector{Number}, xlc::Vector{Number})::NamedTuple{(:gna, :gk, :gs, :gl, :gna_step, :gk_step, :gs_step, :gl_step, :step), Tuple{Number, Number, Number, Number, Number, Number, Number, Number, Number}}
    par = p
    par = @set par.gna = xlc[1]
    par = @set par.gk = xlc[2]
    par = @set par.gs = xlc[3]
    par = @set par.gl = xlc[4]
    par = @set par.gna_step = x[1] - xlc[1]
    par = @set par.gk_step = x[2] - xlc[2]
    par = @set par.gs_step = x[3] - xlc[3]
    par = @set par.gl_step = x[4] - xlc[4]
    return par
end

const use_continuation = true
const use_fast_ode = true
verbose = 2
dataTime = 50000.0 # 50000.0 for final results
if use_continuation
    println("Using continuation")
    paramMap(x,y) = param_map(x,y)
    const p = Model.params_cont
    prob = ODEProblem(Model.ode_cont!, Model.ic, (0.0, dataTime), Model.params_cont, abstol=1e-10, reltol=1e-8, maxiters=1e7)
    # Set up continuation solver
    lens = (@lens _.step)
    const bp = BifurcationProblem(Model.ode_cont!, Model.ic_conv, Model.params_cont, lens)
    solver(v, w, x, y, z, verbose) = contSolver(v, w, x, y, z, bp, verbose)
    const opts_br = ContinuationPar(p_min = 0.0, p_max = 1.0, max_steps = 150, tol_stability = 1e-8, ds=1.0, dsmax=1.0, 
    dsmin=1e-6, detect_bifurcation=0, detect_fold=false,)
else
    println("Using ODE solver")
    paramMap(x, _) = param_map(x)
    const p = Model.params
    prob = ODEProblem(Model.ode!, Model.ic, (0.0, dataTime), Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e7)
    if use_fast_ode
        println("Using fast ODE solver")
        solver = odeSolverCheap
    else
        println("Using full ODE solver")
        solver = odeSolverFull
    end
end

# Parameters we will try to fit to
pTrue = p
pTrue = @set pTrue.gna = 110.0
pTrue = @set pTrue.gk = 11.0
pTrue = @set pTrue.gs = 12.0
pTrue = @set pTrue.gl = 0.25
# Create data
#   Run ODE to converged limit cycle
prob_true = remake(prob, p=pTrue)::ODEProblem
sol = solve(prob_true, Tsit5())::ODESolution
display(plot(sol, idxs=Model.slow_idx, title="Check limit cycle is converged for true data"))
#   Generate aligned data
sol_pulse, period = aligned_sol(sol[end], prob_true,)
#   Add noise and plot
odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))
plot(sol_pulse, title="True data"; label="Simulation")
display(plot!(sol_pulse.t, odedata, label="Data"))

println("Log likelihood of true parameters: ", ll(sol.u[end], odedata, 2.0, prob_true))
numSamples = 1000*5*10 # 1000 samples per parameter before adaption (10% of the samples)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 100

chain, accepts = mcmc(numSamples, solver, [120.0, 13.0, 10.0, 0.3, 1.5], prob, odedata, paramMap, verbose)

display(plot([mean(accepts[1:i]) for i in 1:numSamples], title="Acceptance rate", xlabel="Iteration", ylabel="Acceptance rate"))

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
