using DifferentialEquations
using Revise, Parameters, Plots
using BifurcationKit
using LinearAlgebra
const BK = BifurcationKit

function pow(a, b)
    return a^b
end

function sqr(a)
    return a^2
end

function ln(a)
    return log(a)
end

function LR!(du, u, p, t=0.0)
    @unpack Am, V_myo, V_JSR, V_NSR, R, T, F, Cm, g_Na, g_Nab, g_Cab, g_K1_max, g_Kp, g_K_max, Nao, Cao, Ko, gamma_Nai, gamma_Nao, gamma_Ki, gamma_Ko, P_Ca, 
    P_Na, P_K, gamma_Cai, gamma_Cao, Km_Ca, PR_NaK, K_mpCa, I_pCa, I_NaK, K_mNai, K_mKo, P_ns_Ca, K_m_ns_Ca, K_NaCa, K_mNa, K_mCa, K_sat, eta, G_rel_max, 
    tau_on, tau_off, tau_tr, K_mrel, K_mup, I_up, Ca_NSR_max, delta_Ca_ith, delta_Ca_i2, t_CICR = p
    cost, sint, m, h, j, d, f, X, V, Nai, Cai, Ca_JSR, Ca_NSR, Ki = u

    ω = 2pi/pulse_period
    x2 = cost^2 + sint^2
    du[1] = cost - ω*sint - cost*x2
    du[2] = sint + ω*cost - sint*x2

    IStimC = I_st(cost, sint)

    # Fast sodium
    alpha_m = 0.32*(V+47.13)/(1.0-exp(-0.1*(V+47.13)))
    beta_m = 0.08*exp(-V/11.0)
    E_Na = R*T/F*ln(Nao/Nai)
    i_Na = g_Na*pow(m, 3.0)*h*j*(V-E_Na)
    if V < -40.0
        alpha_h = 0.135*exp((80.0+V)/-6.8)
        beta_h = 3.56*exp(0.079*V)+310000.0*exp(0.35*V)
        alpha_j = (-127140.0*exp(0.2444*V)-0.00003474*exp(-0.04391*V))*(V+37.78)/(1.0+exp(0.311*(V+79.23)))
        beta_j = 0.1212*exp(-0.01052*V)/(1.0+exp(-0.1378*(V+40.14)))
    else
        alpha_h = 0.0
        beta_h = 1.0/(0.13*(1.0+exp((V+10.66)/-11.1)))
        alpha_j = 0.0
        beta_j = 0.3*exp(-0.0000002535*V)/(1.0+exp(-0.1*(V+32.0)))
    end

    # L-type calcium
    f_Ca = 1.0/(1.0+sqr(Cai/Km_Ca))
    I_CaCa = P_Ca*sqr(2.0)*V*sqr(F)/(R*T)*(gamma_Cai*Cai*exp(2.0*V*F/(R*T))-gamma_Cao*Cao)/(exp(2.0*V*F/(R*T))-1.0)
    I_CaNa = P_Na*sqr(1.0)*V*sqr(F)/(R*T)*(gamma_Nai*Nai*exp(1.0*V*F/(R*T))-gamma_Nao*Nao)/(exp(1.0*V*F/(R*T))-1.0)
    I_CaK = P_K*sqr(1.0)*V*sqr(F)/(R*T)*(gamma_Ki*Ki*exp(1.0*V*F/(R*T))-gamma_Ko*Ko)/(exp(1.0*V*F/(R*T))-1.0)
    i_CaCa = d*f*f_Ca*I_CaCa
    i_CaNa = d*f*f_Ca*I_CaNa
    i_CaK = d*f*f_Ca*I_CaK
    i_Ca_L = i_CaCa+i_CaK+i_CaNa

    d_infinity = 1.0/(1.0+exp(-(V+10.0)/6.24))
    tau_d = d_infinity*(1.0-exp(-(V+10.0)/6.24))/(0.035*(V+10.0))
    alpha_d = d_infinity/tau_d
    beta_d = (1.0-d_infinity)/tau_d
    
    f_infinity = 1.0/(1.0+exp((V+35.06)/8.6))+0.6/(1.0+exp((50.0-V)/20.0))
    tau_f = 1.0/(0.0197*exp(-sqr(0.0337*(V+10.0)))+0.02)
    alpha_f = f_infinity/tau_f
    beta_f = (1.0-f_infinity)/tau_f

    # Time dependent potassium
    Xi = 1.0/(1.0+exp((V-56.26)/32.1))
    g_K = g_K_max*sqrt(Ko/5.4)
    E_K = R*T/F*ln((Ko+PR_NaK*Nao)/(Ki+PR_NaK*Nai))
    i_K = g_K*sqr(X)*Xi*(V-E_K)

    alpha_X = 0.0000719*(V+30.0)/(1.0-exp(-0.148*(V+30.0)))
    beta_X = 0.000131*(V+30.0)/(-1.0+exp(0.0687*(V+30.0)))

    # Time independent potassium
    E_K1 = R*T/F*ln(Ko/Ki)
    alpha_K1 = 1.02/(1.0+exp(0.2385*(V-E_K1-59.215)))
    beta_K1 = (0.49124*exp(0.08032*(V+5.476-E_K1))+exp(0.06175*(V-(E_K1+594.31))))/(1.0+exp(-0.5143*(V-E_K1+4.753)))
    K1_infinity = alpha_K1/(alpha_K1+beta_K1)
    g_K1 = g_K1_max*sqrt(Ko/5.4)
    i_K1 = g_K1*K1_infinity*(V-E_K1)

    # Plateau potassium
    E_Kp = E_K1
    Kp = 1.0/(1.0+exp((7.488-V)/5.98))
    i_Kp = g_Kp*Kp*(V-E_Kp)

    # Sarcolemmal calcium pump
    i_p_Ca = I_pCa*Cai/(K_mpCa+Cai)

    # Sodium background
    E_NaN = E_Na
    i_Na_b = g_Nab*(V-E_NaN)

    # Calcium background
    E_CaN = R*T/(2.0*F)*ln(Cao/Cai)
    i_Ca_b = g_Cab*(V-E_CaN)

    # Sodium potassium pump
    sigma = 1.0/7.0*(exp(Nao/67.3)-1.0)
    f_NaK = 1.0/(1.0+0.1245*exp(-0.1*V*F/(R*T))+0.0365*sigma*exp(-V*F/(R*T)))
    i_NaK = I_NaK*f_NaK*1.0/(1.0+pow(K_mNai/Nai, 1.5))*Ko/(Ko+K_mKo)

    # Non-specific calcium activated current
    EnsCa = R*T/F*ln((Ko+Nao)/(Ki+Nai))
    Vns = V-EnsCa
    I_ns_Na = P_ns_Ca*sqr(1.0)*Vns*sqr(F)/(R*T)*(gamma_Nai*Nai*exp(1.0*Vns*F/(R*T))-gamma_Nao*Nao)/(exp(1.0*Vns*F/(R*T))-1.0)
    I_ns_K = P_ns_Ca*sqr(1.0)*Vns*sqr(F)/(R*T)*(gamma_Ki*Ki*exp(1.0*Vns*F/(R*T))-gamma_Ko*Ko)/(exp(1.0*Vns*F/(R*T))-1.0)
    i_ns_Na = I_ns_Na*1.0/(1.0+pow(K_m_ns_Ca/Cai, 3.0))
    i_ns_K = I_ns_K*1.0/(1.0+pow(K_m_ns_Ca/Cai, 3.0))
    i_ns_Ca = i_ns_Na+i_ns_K
    
    # Sodium calcium exchanger
    i_NaCa = K_NaCa*1.0/(pow(K_mNa, 3.0)+pow(Nao, 3.0))*1.0/(K_mCa+Cao)*1.0/(1.0+K_sat*exp((eta-1.0)*V*F/(R*T)))*(exp(eta*V*F/(R*T))*pow(Nai, 3.0)*Cao-exp((eta-1.0)*V*F/(R*T))*pow(Nao, 3.0)*Cai)

    # Calcium fluxes in the SR
    if delta_Ca_i2 < delta_Ca_ith
        G_rel_peak = 0.0
    else
        G_rel_peak = G_rel_max
    end
    G_rel = G_rel_peak*(delta_Ca_i2-delta_Ca_ith)/(K_mrel+delta_Ca_i2-delta_Ca_ith)*(1.0-exp(-t_CICR/tau_on))*exp(-t_CICR/tau_off)
    i_rel = G_rel*(Ca_JSR-Cai)

    i_up = I_up*Cai/(Cai+K_mup)
    K_leak = I_up/Ca_NSR_max
    i_leak = K_leak*Ca_NSR
    i_tr = (Ca_NSR-Ca_JSR)/tau_tr

    du[3] = alpha_m*(1.0-m)-beta_m*m
    du[4] = alpha_h*(1.0-h)-beta_h*h
    du[5] = alpha_j*(1.0-j)-beta_j*j
    du[6] = alpha_d*(1.0-d)-beta_d*d
    du[7] = alpha_f*(1.0-f)-beta_f*f
    du[8] = alpha_X*(1.0-X)-beta_X*X
    du[9] = (IStimC-(i_Na+i_Ca_L+i_K+i_K1+i_Kp+i_NaCa+i_p_Ca+i_Na_b+i_Ca_b+i_NaK+i_ns_Ca))/Cm
    du[10] = -(i_Na+i_CaNa+i_Na_b+i_ns_Na+i_NaCa*3.0+i_NaK*3.0)*Am/(V_myo*F)
    du[11] = -(i_CaCa+i_p_Ca+i_Ca_b-i_NaCa)*Am/(2.0*V_myo*F)+i_rel*V_JSR/V_myo+(i_leak-i_up)*V_NSR/V_myo
    du[12] = -(i_rel-i_tr*V_NSR/V_JSR)
    du[13] = -(i_leak+i_tr-i_up)
    du[14] = -(i_CaK+i_K+i_K1+i_Kp+i_ns_K+-i_NaK*2.0)*Am/(V_myo*F)
end

function LR_time_free!(du, u, p)
    LR!(du, u, p, 0.0)
end

m = 0.0
h = 1.0
j = 1.0
d = 0.0
f = 1.0
X = 0.0
V = -84.624
Nai = 10.0
Cai = 0.12e-3
Ca_JSR = 1.8
Ca_NSR = 1.8
Ki = 145.0
u0 = [m, h, j, d, f, X, V, Nai, Cai, Ca_JSR, Ca_NSR, Ki]

paramLR = (
# Cell geometry
Am = 200, V_myo = 0.68, V_JSR = 0.0048, V_NSR = 0.0552,
# Physical constants
R = 8.3145e3, T = 310.0, F = 96845.0,
# Conductances and capacitance
Cm = 0.01, g_Na = 0.16, g_Nab = 1.41e-5, g_Cab = 3.016e-5, g_K1_max = 7.5e-3, g_Kp = 1.83e-4, g_K_max = 2.82e-3,
# Extracellular concentrations
Nao = 140.0, Cao = 1.8, Ko = 5.4,
# Other parameters
gamma_Nai = 0.75, gamma_Nao = 0.75, gamma_Ki = 0.75, gamma_Ko = 0.75, P_Ca = 5.4e-6, P_Na = 6.75e-9, P_K = 1.93e-9, gamma_Cai = 1.0, gamma_Cao = 0.34, Km_Ca = 0.6e-3,
PR_NaK = 0.01833, K_mpCa = 0.5e-3, I_pCa = 1.15e-2, I_NaK = 1.5e-2, K_mNai = 10.0, K_mKo = 1.5, P_ns_Ca = 1.75e-9, K_m_ns_Ca = 1.2e-3, K_NaCa = 20.0, K_mNa = 87.5,
K_mCa = 1.38, K_sat = 0.1, eta = 0.35, G_rel_max = 60.0, tau_on = 2.0, tau_off = 2.0, tau_tr = 180.0, K_mrel = 0.8e-3, K_mup = 0.92e-3, I_up = 0.005, Ca_NSR_max = 15.0,
delta_Ca_ith = 0.18e-3,
# Missing from original cellml
delta_Ca_i2 = 0.0, t_CICR = 0.0,
)

function tstops(maxt)
    # Make sure a point inside each stimulus pulse is included in ode solve
    t = 0.0
    tstop = []
    while t < maxt
        push!(tstop, t+pulse_width/2)
        t += pulse_period
    end
    return tstop
end

function I_st(t)
    # Stimulus current using time
    if t%pulse_period < pulse_width
        return pulse_amplitude
    else
        return 0.0
    end
end

function I_st(cost, sint)
    # Stimulus current using cos(ωt) and sin(ωt)
    ω = 2pi/pulse_period
    if cost >= cos(ω*pulse_width/2)
        return pulse_amplitude
    else
        return 0.0
    end
end

function convergence_plot(sol, dt=1000)
    # Plot the change in the states across each pulse
    error = []
    for i in dt:dt:sol.t[end]
        push!(error, norm((sol(i)-sol(i-dt))./scaling))
    end
    # Plot error on a log scale
    plot(1:length(error), error, yscale=:log10)
    title!("Convergence plot")
    xlabel!("Pulse count")
    println(error)
    display(ylabel!("Error"))
end

# The following initial conditions have been run to convergence at abstol=1e-13, reltol=1e-11
u0 = [0.0018898062554417226
0.9802447503382736
0.9876401575874676
7.31687558525049e-6
0.9972346826295406
0.0025193256116648756
-83.78994249513194
25.430317134992308
0.0006556590517487814
6.526242185630225
6.524882007576133
136.09843018354334]

maxt = 20000.0
pulse_width = 0.5
pulse_period = 500.0
pulse_amplitude = 1.0

u0 = [cos(-pulse_width*2pi/pulse_period/2), sin(-pulse_width*2pi/pulse_period/2), u0...]

scaling = copy(u0)
scaling[1:6] .= 1.0
id_V = 9

prob_ode = ODEProblem(LR!, u0, (0.0,maxt), paramLR, abstol=1e-12, reltol=1e-10)
sol_ode = solve(prob_ode, Rodas5(), tstops = tstops(maxt), maxiters=1e7);
convergence_plot(sol_ode, pulse_period)

# u0 = sol_ode.u[end]
prob_ode = remake(prob_ode, u0=u0, tspan=(0.0,pulse_period))
sol_pulse = solve(prob_ode, Rodas5(), tstops = tstops(pulse_period), maxiters=1e7)
plot(sol_pulse, idxs=id_V)
title!("Single action potential")
xlabel!("Time (ms)")
display(ylabel!("V (mV)"))

bp = BifurcationProblem(LR!, u0, paramLR, (@lens _[9]); # 9 is gNa
record_from_solution = (x,p) -> V=x[id_V])

argspo = (record_from_solution = (x, p) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[id_V,:]),
				min = minimum(xtt[id_V,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[id_V,:]; label = "V", k...)
	end)

probsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M=1),
bp, prob_ode, sol_pulse, pulse_period; alg=Rodas5(), abstol=1e-12, reltol=1e-10)

opts_br = ContinuationPar(p_min = 0.05, p_max = 0.4, max_steps = 10000, 
    newton_options = NewtonPar(tol=1e-6, verbose=true),
    ds=1e-11, dsmin = 1e-12, dsmax = 1e-9)
brpo_fold = continuation(probsh, cish, PALC(), opts_br;
	verbosity = 3, plot = true, normC = norminf,
	argspo...
)
