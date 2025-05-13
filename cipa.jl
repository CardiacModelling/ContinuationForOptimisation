module Cipa

export ode!, params, ic, ic_conv, plot_idx, slow_idx, isoutofdomain, tstops, param_map!, lcerror

using DifferentialEquations
using Parameters, Plots
using LinearAlgebra, NaNMath
using Accessors
using CellMLToolkit
using SymbolicIndexingInterface


ml = CellModel("ohara_rudy_cipa_v1_2017.cellml")
tspan = (0.0, 1000.0)
prob = ODEProblem(ml, tspan)

pulse_period = 1000.0
pulse_width = 0.5
pulse_start = 10.0

# Update tstops to use ml defined period and width etc
function tstops(maxt)
    # Make sure a point inside each stimulus pulse is included in ode solve
    t = 0.0
    tstop = []
    while t < maxt
        push!(tstop, t + pulse_width / 2 + pulse_start)
        t += pulse_period
    end
    return tstop
end

# The following initial conditions have been run to convergence at abstol=1e-11, reltol=1e-9 over 2000s
ic_conv = [0.012981240825476272, 2.386982419911664e-9, 0.9999999906690992, 0.9999999906673397, 0.9997897035066126, 0.9999999906690588, 0.9999999906672297, 0.9084186800742879, 0.9999721216585953, 0.0028503055123109636, 0.996781265848355, 1.829783419057642e-8, 8.399668294663825e-5, 0.9996302752175523, 6.942766452016685e-5, 5.801622721457484e-5, 0.00015853892449436627, 0.27453089524368957, 0.00019461796420164882, 0.4991949465526101, 0.26956228107712715, 0.000191241458020256, 0.6952364439122444, 0.6952259041909397, 0.451608517183822, 0.6951667300143216, 0.6951268302964935, 0.007405436490075119, 0.0010066907297406323, 0.0005129375080839616, 0.9995477004049124, 0.9995477050548593, 0.5860225118147987, 0.6389933768589123, 8.684711300529053e-5, 1.5920982167118305, 1.6410709479194885, 8.57331187410234e-5, 144.37614089083624, 144.3761099798156, 7.54317077248389, 7.543261377480067, -87.91922419741937, 2.669215233968939e-7, 3.3348127601598776e-7]

ic = prob.u0
params = list_params(ml)
id_V = 43

function solversettings(; save=true, maxt=1000.0)
    if save
        return (abstol=1e-12, reltol=1e-11, maxiters=1e9, tstops=tstops(maxt), tspan=(0.0, maxt))
    else
        return (save_everystep=false, save_start=false, save_end=true,
        abstol=1e-12, reltol=1e-11, maxiters=1e9, tstops=tstops(maxt), tspan=(0.0, maxt))
    end
end

setter! = setp(getsys(ml), [:IKb₊GKb_b, :INa₊GNa, :IKr₊GKr_b, :IK1₊GK1_b, :INaCa_i₊Gncx_b,
                                :INaL₊GNaL_b, :IKs₊GKs_b, :IpCa₊GpCa, :Ito₊Gto_b])

function param_map!(prob, params)
    defaultParams = [0.003, 75.0, 0.04658545454545456, 0.3239783999999998, 0.0008,
                     0.019957499999999975, 0.006358000000000001, 0.0005, 0.02]
    setter!(prob, defaultParams .* params)
    return nothing
end

#TODO Param_map now mutates, want to make sure this doesnt break anything

function lcerror(ic, p)
    param_map!(prob, p)
    prob_de = remake(prob, u0=ic)
    sol = solve(prob_de, Tsit5(); solversettings(save=false, maxt=1000.0)...)
    error = sum(abs.(sol[end] .- ic))
    return error
end

function lcerror_withdx(ic, p)
    param_map!(prob, p)
    prob_de = remake(prob, u0=ic)
    sol = solve(prob_de, Tsit5(); solversettings(save=false, maxt=1000.0)...)
    error = sum(abs.(sol[end] .- ic))
    dx = sol[end] - ic
    return error, dx
end

function getlc_standard(p)
    condition(_, t, _) = (t+pulse_period/2) % pulse_period - pulse_period/2
    STATE::Vector{Float64} = zeros(size(Model.ic))
    function affect!(integrator)
        error = STATE .- integrator.u
        if sum(abs.(error)) < 1e-6
            terminate!(integrator)
        end
        STATE .= integrator.u
    end
    cb = ContinuousCallback(condition, affect!, nothing;
        save_positions=(false, false))
    prob_de = remake(prob, p=param_map(p))
    sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0, 100000.0), maxiters=1e9, save_everystep=false, save_start=false, save_end=true, callback=cb)
    if sol.t[end] == 100000
        throw("maximum time reached without converence")
    end
    return sol.u[end]
end

function getlc_tracking(p, ic)
    condition(_, t, _) = (t+pulse_period/2) % pulse_period - pulse_period/2
    STATE::Vector{Float64} = zeros(size(Model.ic))
    function affect!(integrator)
        error = STATE .- integrator.u
        if sum(abs.(error)) < 1e-6
            terminate!(integrator)
        end
        STATE .= integrator.u
    end
    cb = ContinuousCallback(condition, affect!, nothing;
        save_positions=(false, false))
    prob_de = remake(prob, p=param_map(p))
    prob_de = remake(prob_de, u0=ic)
    sol = DifferentialEquations.solve(prob_de, Tsit5(), tspan=(0, 100000.0), maxiters=1e9, u0=Model.ic_conv, save_everystep=false, save_start=false, save_end=true, callback=cb)
    if sol.t[end] == 100000
        throw("maximum time reached without converence")
    end
    return sol.u[end]
end

function findlc(startlc, p, debug)
    x0 = copy(startlc)
    error, dx = lcerror_withdx(x0, p)
    i = 0
    while error > 1e-6
        improving = true
        k = 1
        while improving && error > 1e-6
            # Proposed point
            xp = x0 .+ dx .* k
            errorp, dxp = lcerror_withdx(xp, p)
            i += 1
            if errorp < error
                x0 = xp
                error = errorp
                dx = dxp
                k *= 2
            else
                improving = false
            end
            debug && println("Iteration: $i, Error: $error, Proposed Error: $errorp, Improved: $improving, k: $(improving ? k/2 : k)")
            if i > 500
                debug && println("Too many iterations, stopping.")
                return nothing
            end
        end
        if improving == false && k == 1
            # If we are not improving and k == 1, we are stuck
            debug && println("Not improving, stopping.")
            return nothing
        end
    end
    return x0
end

function continuation(startlc, startp, endp, debug)
    dp = (endp .- startp)
    p = startp
    lc = startlc
    k = 0 # Step covered so far
    alpha = 1 # Step size
    while lcerror(lc, endp) > 1e-6
        debug && println("Trialing parameter step: $(k+alpha)")
        # Find the next point
        lcp = findlc(lc, startp + dp * min(k + alpha, 1), debug)
        if isnothing(lcp)
            debug && println("Failed to find limit cycle, halving step size.")
            # Halve the step size
            alpha /= 2
            if alpha < 0.01
                debug && println("Step size too small, aborting.")
                throw("Failed to converge to limit cycle")
            end
        else
            lc = lcp
            k += alpha
            debug && println("Found limit cycle for k = $k")
            debug && println(lcerror(lc, endp))
        end
    end
    return lc
end

function getlc_continuation(p, ic, startp)
    return continuation(ic, startp, p, false)
end

plot_idx = id_V
end
