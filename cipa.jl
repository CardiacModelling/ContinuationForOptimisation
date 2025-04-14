module Cipa

export ode!, params, ic, ic_conv, plot_idx, slow_idx, isoutofdomain, tstops, param_map, lcerror

using DifferentialEquations
using Parameters, Plots
using LinearAlgebra, NaNMath
using Accessors
using CellMLToolkit


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
ic_conv = [0.2745308915671682, 0.00019461796427113805, 2.6692152859770975e-7, 3.334812825278298e-7, 0.00019124145813546542, 0.4991949465844819, 0.26956228165609786, 2.3869824217017165e-9, 0.9999999906690588, 0.9084186811078454, 0.9999999906690991, 0.9997897035100992, 0.9999721216593975, 0.9999999906672298, 0.9999999906673397, 0.0028503056506149543, 0.9996302752269318, 6.942765480491881e-5, 1.829783419993994e-8, 8.399668328982747e-5, 0.00015853892459398168, 5.8016227256494223e-5, 0.0, 0.0, 0.0, 0.0, 0.0010066907299559013, 0.9995477004046607, 0.586022512774955, 0.0005129375081937013, 0.9995477050546078, 0.6389933779221231, 0.9967812658491881, 7.5431707840883595, 7.543261389086222, 144.3761408775539, 144.37610996653302, 8.68471141548321e-5, 8.573311982915887e-5, 1.6410709549465399, 1.5920982264117725, 0.012981239873594314, 0.007405436492437041, 0.695236443801819, 0.69522590408039, 0.6951667299031825, 0.4516085170543196, 0.6951268301850645, -87.91922419424762]

ic = prob.u0
params = list_params(ml)
id_V = 49

function solversettings(; save=true, maxt=1000.0)
    if save
        return (abstol=1e-11, reltol=1e-9, maxiters=1e9, tstops=tstops(maxt), tspan=(0.0, maxt))
    else
        return (save_everystep=false, save_start=false, save_end=true,
        abstol=1e-11, reltol=1e-9, maxiters=1e9, tstops=tstops(maxt), tspan=(0.0, maxt))
    end
end

function param_map(p)
    par = copy(params)
    update_list!(par, :IKs₊GKs_b, p[1] * 0.006358000000000001)
    update_list!(par, :INaL₊GNaL_b, p[2] * 0.019957499999999975)
    update_list!(par, :IKr₊GKr_b, p[3] * 0.04658545454545456)
    update_list!(par, :INa₊GNa, p[4]* 75.0)
    # TODO Find some more parameters to fit
    return last.(par)
end

function lcerror(ic, p)
    prob_de = remake(prob, p=param_map(p))
    prob_de = remake(prob_de, u0=ic)
    sol = solve(prob_de, Tsit5(); solversettings(save=false, maxt=1000.0)...)
    error = sum(abs.(sol[end] - ic))
    return error
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
        raise("maximum time reached without converence")
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
        raise("maximum time reached without converence")
    end
    return sol.u[end]
end

plot_idx = id_V
end
