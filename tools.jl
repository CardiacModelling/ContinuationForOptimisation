module Tools
using Distributions, DifferentialEquations, Plots, Accessors

include("./model.jl")
using .Model

# Check all variables are converged automatically
function auto_converge_check(prob::ODEProblem, ic, p::NamedTuple)::Bool
    # Use callbacks to find state at start and end of period (using upcrossings of V=0mV)
    condition(u, _, _) = u[1]
    NUM_TIMES_EFFECT_HIT::Int = 0
    function affect!(integrator)
        NUM_TIMES_EFFECT_HIT += 1 
        if NUM_TIMES_EFFECT_HIT >= 2
            terminate!(integrator)
        end
    end
    cb = ContinuousCallback(condition, affect!, nothing;
    save_positions = (true, false))
    sol = solve(prob, Tsit5(), u0=ic, p=p, tspan=(0.0, 10.0), maxiters=1e9, 
    save_everystep=false, save_start=false, save_end=false, callback=cb)
    error = sol[end] - sol[1]
    return sum(abs.(error))<1e-6
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
    # Use callbacks to find state at start of period (using upcrossings of V=0mV)
    condition(u, _, _) = u[1]
    function affect!(integrator)
        terminate!(integrator)
    end
    cb = ContinuousCallback(condition, affect!, nothing;
    save_positions = (true, false))
    sol = solve(prob, Tsit5(), u0=lc, tspan=(0.0, 10.0), maxiters=1e9, 
    save_everystep=false, save_start=false, save_end=false, callback=cb)
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
