# Metropolis Hastings MCMC to be used for both ODE solver and continuation solver
# Also stores and passes the current limit cycle to the next iteration
using Distributions, LinearAlgebra
using BifurcationKit, DifferentialEquations, BenchmarkTools
using CSV, Tables, Plots

include("./model.jl")
using .Model

include("./tools.jl")
using .Tools

"""
    converge(ic, solver, check, verbose=1)

Converge to a limit cycle using the `solver` function and verify it using `check` functions.

Returns `nothing` if the limit cycle was not found.

# Arguments
- `ic::Vector{Float64}`: The initial conditions to start from.
- `solver::Function`: The function to solve the ODE.
- `check::Function`: The function to check if the limit cycle has converged.
- `verbose::Integer=1`: The verbosity level.

# Returns
- `lc::Vector{Float64}`: The converged limit cycle
"""
function converge(ic::Vector{Float64}, solver::Function, check::Function, verbose::Integer=1)::Union{Vector{Float64}, Nothing}
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
    mcmc(numSamples::Integer, solver::Function, μ₀::Vector{Float64}, prob::ODEProblem, data::Vector{Float64}, paramMap::Function, verbose::Integer=1)

Run an adaptive Metropolis-Hastings MCMC to find the posterior distribution of the parameters.

# Arguments
- `numSamples::Integer`: The number of samples to take.
- `solver::Function`: The function to compute the limit cycle.
- `μ₀::Vector{Float64}`: The initial parameters to start from.
- `prob::ODEProblem`: The ODEProblem for the model.
- `data::Vector{Float64}`: The data to compare the limit cycle to.
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose::Integer=1`: The verbosity level (0=None, 1=Minimal, 2=Standard, 3=Debug).

# Returns
- `chain::Matrix{Number}`: The chain of parameters.
- `accepts::Vector{Float64}`: The acceptance rate of the proposals.
"""
function mcmc(numSamples::Integer, solver::Function, μ₀::Vector{Float64}, prob::ODEProblem, data::Vector{Float64}, paramMap::Function, verbose::Integer=1)
    # Set up and preallocate variables
    chain = zeros(numSamples, length(μ₀))
    accepts = zeros(numSamples)
    x = copy(μ₀)
    σ = x[end]
    a = 1.0
    adaptionStart = ceil(Int, numSamples*0.1) # Start adaptive covariance after 10% of samples
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
        # Solve with new parameters and get log likelihood
        lcNew = converge(lc, (ic) -> solver(xNew, prob, ic, x, paramMap, verbose), (ic) -> Tools.auto_converge_check(prob, ic, paramMap(xNew, xNew)), verbose)
        if lcNew === nothing
            lcNew = lc
            llNew = -Inf
        else
            llNew = ll(lcNew, data, σNew, remake(prob, p=paramMap(xNew, xNew))::ODEProblem, verbose)
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
            println("Local acceptance rate: ", sum(accepts[i-99:i]), "%")
        end
        chain[i, :] = x
        # Adapt the proposal distribution
        if i == adaptionStart + 1 && verbose > 0
            println("Adaption started")
        end
        if i > adaptionStart
            s = i - adaptionStart
            γ = (s+1)^-0.6
            Σ = Hermitian((1-γ)*Σ + γ*(x - μ₀)*(x - μ₀)')
            μ₀ = (1-γ)*μ₀ + γ*x
            a *= exp(γ*(accepts[i] - 0.25))
            if verbose > 0
                println("Current acceptance rate: ", sum(accepts)/i*100, "%")
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
    q(x::Vector{Float64}, Σ::Hermitian{Float64})::Vector{Float64}

Perturb state `x` to get a new state `xNew` and return it.

# Arguments
- `x::Vector{Float64}`: The current state.
- `Σ::Hermitian{Float64}`: The covariance matrix.

# Returns
- `xNew::Vector{Float64}`: The new state.
"""
function q(x::Vector{Float64}, Σ::Hermitian{Float64})::Vector{Float64}
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
    π(x::Vector{Float64})::Number

Calculate the prior log probability density function of `x`.

Uniform priors for conductance parameters - can ignore normalisation constant.
Inverse Gamma prior for noise `σ`.

# Arguments
- `x::Vector{Float64}`: The conductance parameters (uniform prior so ignored), and the noise parameter σ (`InverseGamma(2,3)` distribution).

# Returns
- `π::Number`: The prior logpdf.
"""
function π(x::Vector{Float64})::Number
    ig = InverseGamma(2, 3)
    return logpdf(ig, x[end])
end

"""
    ll(limitCycle::Vector{Float64}, data::Vector{Float64}, σ::Number, prob::ODEProblem, verbose = 1)::Number

Calculate the log-likelihood of the limit cycle compared with the data, and σ.

# Arguments
- `limitCycle::Vector{Float64}`: A point on the limit cycle to compare with the data.
- `data::Vector{Float64}`: The data to compare with the limit cycle.
- `σ::Number`: The estimated noise standard deviation.
- `prob::ODEProblem`: The ODEProblem for the model.
- `verbose::Integer=1`: The verbosity level.

# Returns
- `ll::Number`: The log-likelihood.
"""
function ll(limitCycle::Vector{Float64}, data::Vector{Float64}, σ::Number, prob::ODEProblem, verbose = 1)::Number
    # Get estimate of data using parameters from p and the limit cycle
    sol, = aligned_sol(limitCycle, prob, period)
    # Calculate the log-likelihood of the data
    n = Normal(0, σ)
    if verbose > 2
        plot(sol, label="Proposed")
        plot!(sol.t, data, label="Data")
    end
    return loglikelihood(n, data - sol.u)
end

"""
    aligned_sol(lc::Vector{Float64}, prob::ODEProblem, period::Number = 0.0; save_only_V::Bool = true)

Align the limit cycle in the solution to start at the max of V and fixes the timesteps for recording the data.

# Arguments
- `lc::Vector{Float64}`: The limit cycle to align.
- `prob::ODEProblem`: The ODEProblem for the model.
- `period::Number`: The period of the data.
- `save_only_V::Bool=true`: Save only the V (voltage) variable.

# Returns
- `sol::ODESolution`: The aligned solution.
- `period::Number`: The period of the limit cycle.
"""
function aligned_sol(lc::Vector{Float64}, prob::ODEProblem, period::Number; save_only_V::Bool = true)
    # Simulation of length 2*period to find the max of V
    sol = DifferentialEquations.solve(prob, Tsit5(); tspan=(0.0, period*2.0), u0=lc, save_idxs=Model.plot_idx, saveat=0.01, dense=false)::ODESolution
    # Find the time where V is maximised
    t = sol.t[argmax(sol.u)]
    # Find the states at that time
    sol = DifferentialEquations.solve(prob, Tsit5(); tspan = (0.0,t), u0=sol.prob.u0, save_everystep=false, save_start=false)
    # Get the aligned solution
    if save_only_V
        return DifferentialEquations.solve(prob, Tsit5(), saveat=0.1, save_idxs=Model.plot_idx, tspan=(0.0, period), u0=sol.u[end])::ODESolution, period
    else
        return DifferentialEquations.solve(prob, Tsit5(), saveat=0.1, tspan=(0.0, period), u0=sol.u[end])::ODESolution, period
    end
end

"""
    odeSolverFull(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, verbose::Integer)::Vector{Float64}

Solve the ODE until convergence starting from the default initial conditions.

# Arguments
- `x::Vector{Float64}`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve.
- `::Vector{Float64}`: The previous limit cycle (unused).
- `::Vector{Float64}`: The parameters of the previous limit cycle (unused).
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc::Vector{Float64}`: The converged limit cycle.
"""
function odeSolverFull(x::Vector{Float64}, prob::ODEProblem, ::Vector{Float64}, ::Vector{Float64}, paramMap::Function, verbose=1::Integer)::Vector{Float64}
    tmp = DifferentialEquations.solve(prob, Tsit5(), save_everystep = false; tspan=(0.0, 50000.0), p=paramMap(x, x), save_start=false)::ODESolution
    return tmp[end]
end

"""
    odeSolverCheap(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, verbose::Integer)::Vector{Float64}

Solve the ODE until convergence but starting from the previous limit cycle.

# Arguments
- `x::Vector{Float64}`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve.
- `lc::Vector{Float64}`: The previous limit cycle.
- `::Vector{Float64}`: The parameters of the previous limit cycle (unused).
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc::Vector{Float64}`: The converged limit cycle.
"""
function odeSolverCheap(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, ::Vector{Float64}, paramMap::Function, verbose::Integer)::Vector{Float64}
    tmp = DifferentialEquations.solve(prob, Tsit5(), save_everystep = false; tspan=(0.0, 10000.0), p=paramMap(x, x), u0=lc, save_start=false)::ODESolution
    return tmp[end]
end

"""
    contSolver(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, bp::BifurcationProblem, verbose::Integer)::Vector{Float64}

Perform continuation on the ODE to get the limit cycle.

# Arguments
- `x::Vector{Float64}`: The parameters to find the limit cycle for.
- `prob::ODEProblem`: The ODEProblem to solve during continuation.
- `lc::Vector{Float64}`: The previous limit cycle.
- `xlc::Vector{Float64}`: The parameters of the previous limit cycle.
- `paramMap::Function`: The function to map the parameters from a `Vector` to a `NamedTuple`.
- `bp::BifurcationProblem`: The BifurcationProblem to solve during continuation.
- `verbose=1::Integer`: The verbosity level.

# Returns
- `lc::Vector{Float64}`: The converged limit cycle.
"""
function contSolver(x::Vector{Float64}, prob::ODEProblem, lc::Vector{Float64}, xlc::Vector{Float64}, paramMap::Function, bp::BifurcationProblem, verbose::Integer)::Vector{Float64}
    # Remake BP and prob with the new parameters and limit cycle initial condition
    bp = re_make(bp; u0=lc, params=paramMap(x, xlc))::BifurcationProblem
    prob = remake(prob, u0=lc, p=paramMap(x, xlc))::ODEProblem
    # Create a solution using the previous limit cycle
    sol = DifferentialEquations.solve(prob, Tsit5(), tspan=(0.0, 50.0))::ODESolution
    # Get the shooting problem
    bpsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
    bp, prob, sol, period; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

    # If we actually need to do continuation (rather than just refining the limit cycle), then we need to check both sides
    bothside = xlc != x
    # Catch any errors from BifurcationKit and fall back to ODE solver
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
    # Check if the continuation was successful (the parameter step was 1), if not fall back to ODE solver
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
    sol = DifferentialEquations.solve(prob, Tsit5(), tspan=(0.0, 50000.0), u0=lc)::ODESolution
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

"""
    param_map(x::Vector{Float64})::NamedTuple{(:gna, :gk, :gs, :gl), Tuple{Number, Number, Number, Number}}

Map the parameters from a `Vector` to a `NamedTuple`.

# Arguments
- `x::Vector{Float64}`: The parameters in a `Vector` ordered as gna, gk, gs, gl. The noise parameter can be optionally included at the end.

# Returns
- `par::NamedTuple`: The parameters as a `NamedTuple`.
"""
function param_map(x::Vector{Float64})::NamedTuple{(:gna, :gk, :gs, :gl), Tuple{Number, Number, Number, Number}}
    # Load global const parameters p
    par = p
    # Set the parameters from the state x
    par = @set par.gna = x[1]
    par = @set par.gk = x[2]
    par = @set par.gs = x[3]
    par = @set par.gl = x[4]
    return par
end

"""
    param_map(x::Vector{Float64}, xlc::Vector{Float64})::NamedTuple{(:gna, :gk, :gs, :gl, :gna_step, :gk_step, :gs_step, :gl_step, :step), Tuple{Number, Number, Number, Number, Number, Number, Number, Number, Number}}

Map the parameters from a `Vector` to a `NamedTuple`.

Specific to the continuation solver.

# Arguments
- `x::Vector{Float64}`: The parameters to find the limit cycle for.
- `xlc::Vector{Float64}`: The parameters of the previous limit cycle.

# Returns
- `par::NamedTuple`: The parameters as a `NamedTuple`.
"""
function param_map(x::Vector{Float64}, xlc::Vector{Float64})::NamedTuple{(:gna, :gk, :gs, :gl, :gna_step, :gk_step, :gs_step, :gl_step, :step), Tuple{Number, Number, Number, Number, Number, Number, Number, Number, Number}}
    # Load global const parameters p
    par = p
    # Set the parameters from the state xlc (starting location for continuation)
    par = @set par.gna = xlc[1]
    par = @set par.gk = xlc[2]
    par = @set par.gs = xlc[3]
    par = @set par.gl = xlc[4]
    # Set the continuation step sizes
    par = @set par.gna_step = x[1] - xlc[1]
    par = @set par.gk_step = x[2] - xlc[2]
    par = @set par.gs_step = x[3] - xlc[3]
    par = @set par.gl_step = x[4] - xlc[4]
    return par
end

# Method selection and settings
const use_continuation = true
const use_fast_ode = true
file_type = use_continuation ? "cont_" : (use_fast_ode ? "fastODE_" : "fullODE")
verbose = 2
# Time to run the ODE for the data
dataTime = 50000.0
# Define the method specific settings and functions for MCMC
if use_continuation
    println("Using continuation")
    paramMap(x,y) = param_map(x,y)
    const p = Model.params_cont
    prob = ODEProblem(Model.ode_cont!, Model.ic, (0.0, dataTime), Model.params_cont, abstol=1e-10, reltol=1e-8, maxiters=1e7)
    # Set up continuation solver
    lens = @optic _.step
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

# Create the true data
# True parameters
pTrue = p
pTrue = @set pTrue.gna = 110.0
pTrue = @set pTrue.gk = 11.0
pTrue = @set pTrue.gs = 12.0
pTrue = @set pTrue.gl = 0.25

# Run ODE to converged limit cycle
prob_true = remake(prob, p=pTrue)::ODEProblem
sol = DifferentialEquations.solve(prob_true, Tsit5())::ODESolution
plot(sol, idxs=Model.slow_idx, title="Check limit cycle is converged for true data")
savefig(file_type*"check_converged_mcmc.pdf")
if Tools.auto_converge_check(prob_true, sol[end], pTrue)
    println("Data is appropriately converged")
else
    println("Data was NOT generated from a converged limit cycle")
end

# Generate aligned data
const period = get_period(sol[end], prob_true)
sol_pulse, _ = aligned_sol(sol[end], prob_true, period)
# Add noise and plot
odedata = Array(sol_pulse.u) + 2.0 * randn(size(sol_pulse))
plot(sol_pulse, title="True data"; label="Simulation")
display(plot!(sol_pulse.t, odedata, label="Data"))

# Check the log likelihood of the true parameters
println("Log likelihood of true parameters: ", ll(sol.u[end], odedata, 2.0, prob_true))

# Run MCMC
numSamples = 1000*5*10 # 1000 samples per parameter before adaption (10% of the samples)
numSamples=100
chain, accepts = mcmc(numSamples, solver, [120.0, 13.0, 10.0, 0.3, 1.5], prob, odedata, paramMap, verbose)

# Plot results
plot_params = (linewidth=2., dpi=300, size=(450,300))

# Plot acceptance rate
plot([mean(accepts[1:i]) for i in 1:numSamples], title="Acceptance Rate", xlabel="Iteration", ylabel="Cumulative Acceptance Rate",
ylim=(0,1), label="Acceptance Rate", xlim = (1,numSamples); plot_params...)
vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot)
savefig(file_type*"acceptance.pdf")

# Remove burn in stage to get posterior distribution
burnIn = round(Int, numSamples*0.25)
posterior = chain[burnIn+1:end, :]

# Plot posterior histograms
paramNames = ["gNa" "gK" "gS" "gL" "σ"]
pTrueWithNoise = [Model.params..., 2.0]
for i in axes(posterior, 2)
    histogram(posterior[:, i], normalize=:pdf, title = "Posterior: "*paramNames[i], ylabel = "P(x)",
    legend = false; plot_params...)
    vline!([pTrueWithNoise[i]], color=:black, linewidth=1.5)
    savefig(file_type*"posterior-"*paramNames[i]*".pdf")
end

# Plot parameter convergence
plot((chain'./pTrueWithNoise)', label=paramNames, title="Parameter Convergence", 
xlabel="Iteration", ylabel="Parameter Value (Relative to Truth)", xlim=(1,numSamples); 
plot_params...)
hline!([1.0], label="Truth", color=:black, linewidth=1.5)
vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot)
savefig(file_type*"convergence.pdf")

# Write data to CSV
tab = Tables.table([chain convert(Vector{Bool}, accepts)]; header=[paramNames..., "Accept"])
CSV.write(file_type*"chain.csv", tab)

# Benchmark the MCMC
b = @benchmarkable mcmc($numSamples, $solver, [120.0, 13.0, 10.0, 0.3, 1.5], $prob, $odedata, $paramMap, $verbose)
t = run(b, seconds=120)

Benchmark.save(file_type*"mcmc_benchmark.json", t)
