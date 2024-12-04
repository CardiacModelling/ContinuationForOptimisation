# Metropolis Hastings MCMC to be used for both ODE solver and continuation solver
# Also stores and passes the current limit cycle to the next iteration
using Distributions, LinearAlgebra, Random
using BifurcationKit, DifferentialEquations
using CSV, Tables, Plots, DataFrames, Measures, JLD2, Dates

include("./model.jl")
using .Model

include("./tools.jl")
using .Tools

# Set seed
Random.seed!(1)

"""
    converge(ic, solver, check, verbose=1)

Converge to a limit cycle using the `solver` function and verify it using `check` functions.

Returns `nothing` if the limit cycle was not found.

# Arguments
- `ic`: The initial conditions to start from.
- `solver::Function`: The function to solve the ODE.
- `check::Function`: The function to check if the limit cycle has converged.
- `verbose::Integer=1`: The verbosity level.

# Returns
- `lc`: The converged limit cycle
"""
function converge(ic, solver::Function, check::Function, verbose::Integer=1)::Union{Vector{Float64}, Nothing}
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
    mcmc(numSamples::Integer, solver::Function, μ₀, prob::ODEProblem, data, paramMap::Function, verbose::Integer=1)

Run an adaptive Metropolis-Hastings MCMC to find the posterior distribution of the parameters.

# Arguments
- `numSamples::Integer`: The number of samples to take.
- `solver::Function`: The function to compute the limit cycle.
- `μ₀`: The initial parameters to start from.
- `prob::ODEProblem`: The ODEProblem for the model.
- `data`: The data to compare the limit cycle to.
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `start::Tuple=()`: Start MCMC from a previous iteration (defaults to empty tuple to start fresh). Supply a tuple of the previous iterations data from JLD2.
- `record::Boolean`=true: Save the current state to a JLD2 file and append results to a CSV.
- `verbose::Integer=1`: The verbosity level (0=None, 1=Minimal, 2=Standard, 3=Debug).
"""
function mcmc(numSamples::Integer, solver::Function, μ₀, prob::ODEProblem, data, paramMap::Function, start::Tuple=(), record::Bool=true, verbose::Integer=1)
    # Set up and preallocate variables
    x = copy(μ₀)
    prob = remake(prob, p=paramMap(x, x), u0=Model.ic_conv)::ODEProblem
    adaptionStart = ceil(Int, numSamples*0.1) # Start adaptive covariance after 10% of samples
    if isempty(start)
        chain = zeros(numSamples, length(μ₀))
        accepts = zeros(numSamples)
        lls = zeros(numSamples)
        σ = x[end]
        a = 1.0
        Σ = Hermitian(diagm(μ₀/1e6))
        lc = converge(Model.ic_conv, (ic) -> solver(x, prob, ic, x, paramMap, verbose), (ic) -> Tools.auto_converge_check(prob, ic, paramMap(x, x)), verbose)
        if lc === nothing
            lc = Model.ic_conv
            llOld = -Inf
        else
            llOld = ll(lc, data, σ, prob)
        end
        iteration = 0 # Current iteration
        if verbose > 0
            println("========        Starting MCMC        ========")
            println("Initial parameters: ", x[1:end-1])
            println("Initial noise: ", σ)
            println("Initial limit cycle: ", lc)
            println("Number of samples: ", numSamples)
            println("=============================================")
        end
    else
        iteration, x, lc, llOld, a, Σ, μ₀ = start
        σ = x[end]
        if verbose > 0
            println("======== Continuing MCMC from previous state ========")
            println("Current parameters: ", x[1:end-1])
            println("Current noise: ", σ)
            println("Current limit cycle: ", lc)
            println("Number of samples: ", numSamples)
            println("====================================================")
        end
    end
    # Iterate through MCMC steps
    for i in iteration+1:numSamples # Check this order of operations
        if verbose > 0
            println("========         Iteration ", i, "         ========")
            println("Current parameters: ", x[1:end-1])
            println("Current noise: ", σ)
            println("Current limit cycle: ", lc)
        end
        # Sample from proposal
        xNew = q(x, a*Σ)
        σNew = xNew[end]
        # Solve with new parameters and get log likelihood
        lcNew = converge(lc, (ic) -> solver(xNew, prob, ic, x, paramMap, verbose), (ic) -> Tools.auto_converge_check(prob, ic, paramMap(xNew, xNew)), verbose)
        if lcNew === nothing
            lcNew = lc
            llNew = -Inf
        else
            llNew = ll(lcNew, data, σNew, remake(prob, p=paramMap(xNew, xNew))::ODEProblem, verbose)
            if verbose > 2
                sol = Tools.aligned_sol(lc, remake(prob, p=paramMap(x, x)), period)
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
            accept = 1
        else
            if verbose > 0
                println("- Proposal rejected")
            end
            accept = 0
        end
        if verbose > 0 && i > 100
            println("Local acceptance rate: ", sum(accepts[i-99:i]), "%")
        end
        # Adapt the proposal distribution
        if i == adaptionStart + 1 && verbose > 0
            println("Adaption started")
        end
        if i > adaptionStart
            s = i - adaptionStart
            γ = (s+1)^-0.6
            Σ = Hermitian((1-γ)*Σ + γ*(x - μ₀)*(x - μ₀)')
            μ₀ = (1-γ)*μ₀ + γ*x
            a *= exp(γ*(accept - 0.25))
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
        # Update CSV
        if record
            CSV.write(file_type*"chain.csv", DataFrame([i x... llOld accept], :auto), append=true)
            # Store MCMC state
            start = (i, x, lc, llOld, a, Σ, μ₀)
            JLD2.save_object(file_type*"saved_state.jld2", start)
        end
        println(now())
    end
    return chain, accepts, lls
end

"""
    q(x, Σ::Hermitian{Float64})

Perturb state `x` to get a new state `xNew` and return it.

# Arguments
- `x`: The current state.
- `Σ::Hermitian{Float64}`: The covariance matrix.

# Returns
- `xNew`: The new state.
"""
function q(x, Σ::Hermitian{Float64})
    local d
    try
        d = MvNormal(x, Σ)
    catch e
        if e isa PosDefException
            println("Covariance matrix not positive definite")
            println("Matrix: ", Σ)
            eigs = eigvals(Σ)
            println("Eigenvalues: ", eigs)
            Σ += (eps()-minimum(eigs))*I
            println("Adjusted matrix: ", Σ)
            println("Adjusted eigenvalues: ", eigvals(Σ))
            d = MvNormal(x, Σ)
        else
            rethrow(e)
        end
    end
    return rand(d)
end

"""
    π(x)::Number

Calculate the prior log probability density function of `x`.

Uniform priors for conductance parameters - can ignore normalisation constant.
Inverse Gamma prior for noise `σ`.

# Arguments
- `x`: The conductance parameters (uniform prior so ignored), and the noise parameter σ (`InverseGamma(2,3)` distribution).

# Returns
- `π::Number`: The prior logpdf.
"""
function π(x)::Number
    if any(x .< 0)
        return -Inf
    end
    ig = InverseGamma(2, 3)
    return logpdf(ig, x[end])
end

"""
    ll(limitCycle, data, σ::Number, prob::ODEProblem, verbose = 1)::Number

Calculate the log-likelihood of the limit cycle compared with the data, and σ.

# Arguments
- `limitCycle`: A point on the limit cycle to compare with the data.
- `data`: The data to compare with the limit cycle.
- `σ::Number`: The estimated noise standard deviation.
- `prob::ODEProblem`: The ODEProblem for the model.
- `verbose::Integer=1`: The verbosity level.

# Returns
- `ll::Number`: The log-likelihood.
"""
function ll(limitCycle, data, σ::Number, prob::ODEProblem, verbose = 1)::Number
    if σ < 0
        return -Inf
    end
    # Get estimate of data using parameters from p and the limit cycle
    sol = Tools.aligned_sol(limitCycle, prob, period)
    # Calculate the log-likelihood of the data
    n = Normal(0, σ)
    if verbose > 2
        plot(sol, label="Proposed")
        plot!(sol.t, data, label="Data")
    end
    return loglikelihood(n, data - sol.u)
end

"""
    odeSolverFull(x, prob::ODEProblem, lc, xlc, paramMap::Function, verbose::Integer)

Solve the ODE until convergence starting from the default initial conditions.

# Arguments
- `x`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve.
- `_`: The previous limit cycle (unused).
- `_`: The parameters of the previous limit cycle (unused).
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc`: The converged limit cycle.
"""
function odeSolverFull(x, prob::ODEProblem, lc, xlc, paramMap::Function, verbose=1::Integer)
    tmp = DifferentialEquations.solve(prob, Tsit5(), save_everystep = false; tspan=(0.0, 1000.0), p=paramMap(x, x), save_start=false, maxiters=1e9)::ODESolution
    return tmp[end]
end

"""
    odeSolverCheap(x, prob::ODEProblem, lc, xlc, paramMap::Function, verbose::Integer)

Solve the ODE until convergence but starting from the previous limit cycle.

# Arguments
- `x`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve.
- `lc`: The previous limit cycle.
- `_`: The parameters of the previous limit cycle (unused).
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc`: The converged limit cycle.
"""
function odeSolverCheap(x, prob::ODEProblem, lc, _, paramMap::Function, verbose::Integer)
    tmp = DifferentialEquations.solve(prob, Tsit5(), save_everystep = false; tspan=(0.0, 150.0), p=paramMap(x, x), u0=lc, save_start=false, maxiters=1e9)::ODESolution
    return tmp[end]
end

"""
    early_abort((x, f, J, res, iteration, itlinear, options); kwargs...)

Abort Newton iterations in continuation if residuals are too large. This will attempt again with a smaller step size.
"""
function early_abort((x, f, J, res, iteration, itlinear, options); kwargs...)
	if res < 5e2
		return true
	else
		return false
	end
end

"""
    contSolver(x, prob::ODEProblem, lc, xlc, paramMap::Function, bp::BifurcationProblem, verbose::Integer)

Perform continuation on the ODE to get the limit cycle.

# Arguments
- `x`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve during continuation.
- `lc`: The previous limit cycle.
- `xlc`: The parameters of the previous limit cycle.
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `bp::BifurcationProblem`: The BifurcationProblem to solve during continuation.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc`: The converged limit cycle.
"""
function contSolver(x, prob::ODEProblem, lc, xlc, paramMap::Function, bp::BifurcationProblem, verbose::Integer)
    # Remake BP and prob with the new parameters and limit cycle initial condition
    bp = re_make(bp; u0=lc, params=paramMap(x, xlc))::BifurcationProblem
    prob = remake(prob, u0=lc, p=paramMap(x, xlc))::ODEProblem
    # Create a solution using the previous limit cycle
    sol = DifferentialEquations.solve(prob, Tsit5(), tspan=(0.0, period*2), maxiters=1e9)::ODESolution
    # Get the shooting problem
    bpsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
    bp, prob, sol, period; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

    # If we actually need to do continuation (rather than just refining the limit cycle), then we need to check both sides
    bothside = xlc != x
    # Catch any errors from BifurcationKit and fall back to ODE solver
    local brpo_sh::Union{Nothing, ContResult}
    try
        brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
            verbosity = 0, bothside=bothside, callback_newton = early_abort)
    catch e
        if verbose > 0
            println("Continuation failed: ", e)
        end
        return odeSolverCheap(x, prob, lc, xlc, paramMap, verbose)
    end
    # Check if the continuation was successful (the parameter step was 1), if not fall back to ODE solver
    if brpo_sh.sol[end].p == 1.0
        return brpo_sh.sol[end].x[1:end-1]
    elseif brpo_sh.sol[1].p == 1.0
        return brpo_sh.sol[1].x[1:end-1]
    else
        if verbose > 0
            println("First point: ", brpo_sh.sol[1].p)
            println("Last point: ", brpo_sh.sol[end].p)
            println("No cont point had parameters wanted. Falling back to ODE solver")
        end
        return odeSolverCheap(x, prob, lc, xlc, paramMap, verbose)
    end
end

# Method selection and settings
const use_continuation = true
const use_fast_ode = true
const file_type = "results/mcmc/"*(use_continuation ? "cont_" : (use_fast_ode ? "fastODE_" : "fullODE_"))
verbose = 2
const continue_from_previous = false
# Define the method specific settings and functions for MCMC
if use_continuation
    println("Using continuation")
    paramMap(x,y) = Tools.param_map(x,y)
    const p = Model.params_cont
    prob = ODEProblem(Model.ode_cont!, Model.ic_conv, (0.0, 1000.0), Model.params_cont, abstol=1e-10, reltol=1e-8, maxiters=1e7)
    # Set up continuation solver
    lens = @optic _.step
    const bp = BifurcationProblem(Model.ode_cont!, Model.ic_conv, Model.params_cont, lens)
    solver(v, w, x, y, z, verbose) = contSolver(v, w, x, y, z, bp, verbose)
    const opts_br = ContinuationPar(p_min = 0.0, p_max = 1.0, max_steps = 50, tol_stability = 1e-8, ds=1.0, dsmax=1.0, 
    detect_bifurcation=0, detect_fold=false, newton_options=NewtonPar(tol=1e-10))
else
    println("Using ODE solver")
    paramMap(x, _) = Tools.param_map(x)
    const p = Model.params
    prob = ODEProblem(Model.ode!, Model.ic_conv, (0.0, 1000.0), Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e7)
    if use_fast_ode
        println("Using fast ODE solver")
        solver = odeSolverCheap
    else
        println("Using full ODE solver")
        solver = odeSolverFull
    end
end

if continue_from_previous
    start = JLD2.load_object(file_type*"saved_state.jld2") # Continue previous MCMC
else
    start = () # Start MCMC fresh
    # Create new blank csv for chain to be filled into
    CSV.write(file_type*"chain.csv", DataFrame([name => [] for name in ["Iteration", "gNa", "gK", "gL", "σ", "ℓ", "Accept"]]))
end

# Load the data
data = CSV.read("results/mcmc/data.csv", DataFrame, header=false)
t = data[:, 1]
odedata = data[:, 2]
const period = t[end]

initialGuess = [0.9956494280601382, 0.9974120177992098, 0.9807640816954317, 1.5]
# Run MCMC
numSamples = 1000*length(initialGuess)*10 # 1000 samples per parameter before adaption (10% of the samples)
mcmc(100, solver, initialGuess, prob, odedata, paramMap, start, false, verbose)
println("Actual MCMC run starting now")
println(now())
mcmc(numSamples, solver, initialGuess, prob, odedata, paramMap, start, true, verbose)

# Read the chain data from the CSV file
data = CSV.read(file_type*"chain.csv", DataFrame)
accepts = data.Accept
lls = data.ℓ
chain = Matrix([data.gNa data.gK data.gL data.σ])

# Plot results
plot_params = (linewidth=2., dpi=300, size=(450,300), margin=5mm)
paramNames = ["gNa" "gK" "gL" "σ"]

# Plot acceptance rate
plot([mean(accepts[max(i-499,1):i]) for i in 1:numSamples], title="Acceptance Rate", xlabel="Iteration",
ylim=(0,1), label="Acceptance Rate", xlim = (1,numSamples); plot_params...)
vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot)
savefig(file_type*"acceptance.pdf")

# Plot log likelihood
plot(lls, title="Log Likelihood", xlabel="Iteration", ylabel="Log Likelihood", xlim=(1,numSamples); plot_params...)
vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot)
savefig(file_type*"loglikelihood.pdf")

# Remove burn in stage to get posterior distribution
burnIn = round(Int, numSamples*0.25)
posterior = chain[burnIn+1:end, :]

# Plot posterior histograms
pTrueWithNoise = [1.0, 1.0, 1.0, 2.0]
for i in axes(posterior, 2)
    histogram(posterior[:, i], normalize=:pdf, title = "Posterior: "*paramNames[i], ylabel = "P(x)",
    legend = false; plot_params...)
    vline!([pTrueWithNoise[i]], color=:black, linewidth=1.5)
    savefig(file_type*"posterior-"*paramNames[i]*".pdf")
end

# Plot parameter convergence
order = [4, 3, 2, 1]
plot(chain[:,order]./pTrueWithNoise[order]', label=paramNames[order'], title="Parameter Convergence", 
xlabel="Iteration", ylabel="Parameter Value (Relative to Truth)", xlim=(1,numSamples), legend=:topright; 
plot_params...)
vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot)
savefig(file_type*"convergence.pdf")
