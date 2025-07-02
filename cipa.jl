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

# The following initial conditions have been run to convergence at abstol=1e-12, reltol=1e-11 over 2000s
ic_conv = [0.012981240140831053, 2.386982422779725e-9, 0.9999999906690992, 0.9999999906673398, 0.9997897035003475, 0.9999999906690588, 0.9999999906672298, 0.9084186805598866, 0.9999721216572243, 0.0028503057486359987, 0.9967812658497135, 1.8297834205589815e-8, 8.399668347667685e-5, 0.9996302752322659, 6.942764960449668e-5, 5.801622728551766e-5, 0.00015853892466425272, 0.27453089268634856, 0.00019461796431734582, 0.49919494600913794, 0.26956228063667653, 0.00019124145820469758, 0.6952364437354498, 0.6952259040127108, 0.4516085169718973, 0.6951667298281783, 0.6951268301051092, 0.007405436493856367, 0.0010066907300855992, 0.0005129375082598189, 0.9995477004045055, 0.9995477050544527, 0.586022509842377, 0.6389933747426111, 8.684711485006521e-5, 1.5920982182991825, 1.6410709559866812, 8.573312059691214e-5, 144.37614086807199, 144.37610995705117, 7.543170793462494, 7.543261398461598, -87.91922419234179, 2.6692152442096387e-7, 3.3348127727537244e-7]

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
    failedCounter = 0
    local dxp
    while error > 1e-6
        improving = true
        k = 1
        while improving && error > 1e-6
            # Proposed point
            xp = x0 .+ dx .* k
            errorp, dxp = lcerror_withdx(xp, p)
            i += 1
            debug && println("Iteration: $i, Error: $error, Proposed Error: $errorp, Improved: $(errorp < error), k: $k")
            if errorp < error
                x0 = xp
                error = errorp
                dx = dxp
                k *= 2
                failedCounter = 0
            else
                improving = false
            end
            if i > 500
                debug && println("Too many iterations, stopping.")
                return nothing
            end
        end
        if improving == false && k == 1
            failedCounter += 1
            debug && println("Failed to improve after $failedCounter attempts, taking the step anyway and trying again.")
            if failedCounter > 10
                debug && println("Failed to improve after 10 attempts, stopping.")
                return nothing
            end
            x0 = x0 .+ dxp
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
