module Tools
using Distributions, DifferentialEquations, Plots, Accessors

include("./model.jl")
using .Model

# Check all variables are converged automatically
function auto_converge_check(prob::ODEProblem, ic, p::NamedTuple)::Bool
    # Find the average across the first 10s for each state
    sol = solve(prob, Tsit5(), u0=ic, p=p, tspan=(0.0, 10.0), maxiters=1e9) # TODO I think this needs a save every 0.01s type of thing, same for end
    avgs = mean(sol.u)
    # Run for further 80s to try and converge closer
    sol = solve(prob, Tsit5(), u0=sol[end], p=p, tspan=(0.0, 80.0), save_everystep=false, save_start=false, maxiters=1e9)
    # Run for further 10s and get the range of each State
    sol = solve(prob, Tsit5(), u0=sol[end], p=p, tspan=(0.0, 10.0), maxiters=1e9)
    # If avgs is inside the range of the final 10s then it is converged
    sol = stack(sol.u)
    return all(minimum(sol, dims=2) .< avgs .< maximum(sol, dims=2))
end

"""
    aligned_sol(lc, prob::ODEProblem, period::Number; save_only_V::Bool = true)

Align the limit cycle in the solution to start at the max of V and fixes the timesteps for recording the data.

# Arguments
- `lc`: The limit cycle to align.
- `prob::ODEProblem`: The ODEProblem for the model.
- `period::Number`: The period of the data.
- `save_only_V::Bool=true`: Save only the V (voltage) variable.

# Returns
- `sol::ODESolution`: The aligned solution.
"""
function aligned_sol(lc, prob::ODEProblem, period::Number; save_only_V::Bool = true)
    # Simulation of length 2*period to find the max of V
    sol = DifferentialEquations.solve(prob, Tsit5(); tspan=(0.0, period*2.0), u0=lc, save_idxs=Model.plot_idx, saveat=1e-5, dense=false, maxiters=1e9)::ODESolution
    # Find the time where V is maximised
    t = sol.t[argmax(sol.u)]
    # Find the states at that time
    sol = DifferentialEquations.solve(prob, Tsit5(); tspan = (0.0,t), u0=sol.prob.u0, save_everystep=false, save_start=false, maxiters=1e9)
    # Get the aligned solution
    if save_only_V
        return DifferentialEquations.solve(prob, Tsit5(), saveat=0.001, save_idxs=Model.plot_idx, tspan=(0.0, period), u0=sol.u[end], maxiters=1e9)::ODESolution
    else
        return DifferentialEquations.solve(prob, Tsit5(), saveat=0.001, tspan=(0.0, period), u0=sol.u[end], maxiters=1e9)::ODESolution
    end
end

"""
    param_map(x)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf), Tuple{Number, Number, Number}}

Map the parameters from a `Vector` to a `NamedTuple`.

# Arguments
- `x`: The parameters in a `Vector` ordered as gna, gk, gl. The noise parameter can be optionally included at the end.

# Returns
- `par::NamedTuple`: The parameters as a `NamedTuple`.
"""
function param_map(x)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf), Tuple{Number, Number, Number}}
    par = Model.params
    # Set the parameters from the state x
    par = @set par.g_Na_sf = x[1]
    par = @set par.g_K_sf = x[2]
    par = @set par.g_L_sf = x[3]
    return par
end

"""
    param_map(x, xlc)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf, :na_step, :k_step, :l_step, :step), Tuple{Number, Number, Number, Number, Number, Number, Number}}

Map the parameters from a `Vector` to a `NamedTuple`.

Specific to the continuation solver.

# Arguments
- `x`: The parameters to find the limit cycle for.
- `xlc`: The parameters of the previous limit cycle.

# Returns
- `par::NamedTuple`: The parameters as a `NamedTuple`.
"""
function param_map(x, xlc)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf, :na_step, :k_step, :l_step, :step), Tuple{Number, Number, Number, Number, Number, Number, Number}}
    # Set the parameters from the state xlc (starting location for continuation)
    par = Model.params_cont
    par = @set par.g_Na_sf = xlc[1]
    par = @set par.g_K_sf = xlc[2]
    par = @set par.g_L_sf = xlc[3]
    # Set the continuation step sizes
    par = @set par.na_step = x[1] - xlc[1]
    par = @set par.k_step = x[2] - xlc[2]
    par = @set par.l_step = x[3] - xlc[3]
    return par
end

end
