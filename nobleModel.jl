using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots, DifferentialEquations, BifurcationKit
using LinearAlgebra

@mtkmodel NOBLE begin
    @parameters begin
        fast_Na_curr₊g_Na
        Ca_back_curr₊g_Cab
        hyper_act_curr₊g_f_K
        hyper_act_curr₊g_f_Na
        Na_back_curr₊g_Nab
        time_indep_K_curr₊g_K1
        trans_out_curr₊g_to
    end
    @variables begin
        Na_Ca_exchanger₊i_NaCa(t)
        mem₊V(t)
        hyper_act_curr_y_gate₊y(t)
        time_dep_K_curr_x_gate₊x(t)
        trans_out_curr_s_gate₊s(t)
        fast_Na_curr_m_gate₊m(t)
        fast_Na_curr_h_gate₊h(t)
        sec_in_curr_d_gate₊d(t)
        sec_in_curr_f_gate₊f(t)
        sec_in_curr_f2_gate₊f2(t)
        intra_Na_conc₊Nai(t)
        intra_Ca_conc₊Cai(t)
        intra_Ca_conc₊Ca_up(t)
        intra_Ca_conc₊Ca_rel(t)
        intra_Ca_conc₊p(t)
        ext_K_conc₊Kc(t)
        intra_K_conc₊Ki(t)
        Ca_back_curr₊E_Ca(t)
        Ca_back_curr₊i_Ca_b(t)
        ext_K_conc₊i_mK(t)
        fast_Na_curr₊E_mh(t)
        fast_Na_curr₊i_Na(t)
        fast_Na_curr_h_gate₊α_h(t)
        fast_Na_curr_h_gate₊β_h(t)
        fast_Na_curr_m_gate₊E0_m(t)
        fast_Na_curr_m_gate₊α_m(t)
        fast_Na_curr_m_gate₊β_m(t)
        hyper_act_curr₊E_K(t)
        hyper_act_curr₊E_Na(t)
        hyper_act_curr₊i_f(t)
        hyper_act_curr₊i_fK(t)
        hyper_act_curr₊i_fNa(t)
        hyper_act_curr_y_gate₊E0_y(t)
        hyper_act_curr_y_gate₊α_y(t)
        hyper_act_curr_y_gate₊β_y(t)
        intra_Ca_conc₊V_rel(t)
        intra_Ca_conc₊V_up(t)
        intra_Ca_conc₊α_p(t)
        intra_Ca_conc₊β_p(t)
        intra_Ca_conc₊i_rel(t)
        intra_Ca_conc₊i_tr(t)
        intra_Ca_conc₊i_up(t)
        intra_Na_conc₊Vi(t)
        mem₊RTONF(t)
        sec_in_curr₊i_si(t)
        sec_in_curr₊i_siCa(t)
        sec_in_curr₊i_siK(t)
        sec_in_curr₊i_siNa(t)
        sec_in_curr_d_gate₊E0_d(t)
        sec_in_curr_d_gate₊α_d(t)
        sec_in_curr_d_gate₊β_d(t)
        sec_in_curr_f2_gate₊β_f2(t)
        sec_in_curr_f_gate₊E0_f(t)
        sec_in_curr_f_gate₊α_f(t)
        sec_in_curr_f_gate₊β_f(t)
        Na_back_curr₊i_Na_b(t)
        Na_K_pump₊i_p(t)
        time_dep_K_curr₊i_K(t)
        time_dep_K_curr_x_gate₊α_x(t)
        time_dep_K_curr_x_gate₊β_x(t)
        time_indep_K_curr₊i_K1(t)
        trans_out_curr₊i_to(t)
        trans_out_curr_s_gate₊α_s(t)
        trans_out_curr_s_gate₊β_s(t)
    end
    @equations begin
        Na_Ca_exchanger₊i_NaCa ~ 0.02 * (exp(0.5 * (3 - 2) * mem₊V / mem₊RTONF) * intra_Na_conc₊Nai^3 * 2 - exp((0.5 - 1) * (3 - 2) * mem₊V / mem₊RTONF) * 140^3 * intra_Ca_conc₊Cai) / ((1 + 0.001 * (intra_Ca_conc₊Cai * 140^3 + 2 * intra_Na_conc₊Nai^3)) * (1 + intra_Ca_conc₊Cai / 0.0069))
        Ca_back_curr₊E_Ca ~ 0.5 * mem₊RTONF * log(2 / intra_Ca_conc₊Cai)
        Ca_back_curr₊i_Ca_b ~ Ca_back_curr₊g_Cab * (mem₊V - Ca_back_curr₊E_Ca)
        ext_K_conc₊i_mK ~ time_indep_K_curr₊i_K1 + time_dep_K_curr₊i_K + hyper_act_curr₊i_fK + sec_in_curr₊i_siK + trans_out_curr₊i_to - 2 * Na_K_pump₊i_p
        fast_Na_curr₊E_mh ~ mem₊RTONF * log((140 + 0.12 * ext_K_conc₊Kc) / (intra_Na_conc₊Nai + 0.12 * intra_K_conc₊Ki))
        fast_Na_curr₊i_Na ~ fast_Na_curr₊g_Na * fast_Na_curr_m_gate₊m^3 * fast_Na_curr_h_gate₊h * (mem₊V - fast_Na_curr₊E_mh)
        fast_Na_curr_h_gate₊α_h ~ 20  * exp(-0.125  * (mem₊V + 75 ))
        fast_Na_curr_h_gate₊β_h ~ 2000  / (320 * exp(-0.1  * (mem₊V + 75 )) + 1)
        fast_Na_curr_m_gate₊E0_m ~ mem₊V + 41 
        fast_Na_curr_m_gate₊α_m ~ 200  * fast_Na_curr_m_gate₊E0_m / (1 - exp(-0.1  * fast_Na_curr_m_gate₊E0_m))
        fast_Na_curr_m_gate₊β_m ~ 8000  * exp(-0.056  * (mem₊V + 66 ))
        hyper_act_curr₊E_K ~ mem₊RTONF * log(ext_K_conc₊Kc / intra_K_conc₊Ki)
        hyper_act_curr₊E_Na ~ mem₊RTONF * log(140 / intra_Na_conc₊Nai)
        hyper_act_curr₊i_f ~ hyper_act_curr₊i_fNa + hyper_act_curr₊i_fK
        hyper_act_curr₊i_fK ~ hyper_act_curr_y_gate₊y * ext_K_conc₊Kc / (ext_K_conc₊Kc + 45) * hyper_act_curr₊g_f_K * (mem₊V - hyper_act_curr₊E_K)
        hyper_act_curr₊i_fNa ~ hyper_act_curr_y_gate₊y * ext_K_conc₊Kc / (ext_K_conc₊Kc + 45) * hyper_act_curr₊g_f_Na * (mem₊V - hyper_act_curr₊E_Na)
        hyper_act_curr_y_gate₊E0_y ~ mem₊V + 52  - 10 
        hyper_act_curr_y_gate₊α_y ~ 0.05  * exp(-0.067  * (mem₊V + 52  - 10 ))
        hyper_act_curr_y_gate₊β_y ~ 1  * hyper_act_curr_y_gate₊E0_y / (1 - exp(-0.2  * hyper_act_curr_y_gate₊E0_y))
        intra_Ca_conc₊V_rel ~ intra_Na_conc₊Vi * 0.02
        intra_Ca_conc₊V_up ~ intra_Na_conc₊Vi * 0.05
        intra_Ca_conc₊α_p ~ 0.625  * (mem₊V + 34 ) / (exp((mem₊V + 34 ) / 4 ) - 1)
        intra_Ca_conc₊β_p ~ 5  / (1 + exp(-1 * (mem₊V + 34 ) / 4 ))
        intra_Ca_conc₊i_rel ~ 2 * 1e15 * intra_Ca_conc₊V_rel * 9.64853414999999950e4 / (1e6 * 0.05) * intra_Ca_conc₊Ca_rel * intra_Ca_conc₊Cai^2 / (intra_Ca_conc₊Cai^2 + 0.001^2)
        intra_Ca_conc₊i_tr ~ 2 * 1e15 * intra_Ca_conc₊V_rel * 9.64853414999999950e4 / (1e6 * 2) * intra_Ca_conc₊p * (intra_Ca_conc₊Ca_up - intra_Ca_conc₊Ca_rel)
        intra_Ca_conc₊i_up ~ 2 * 1e15 * intra_Na_conc₊Vi * 9.64853414999999950e4 / (1e6 * 0.025 * 5) * intra_Ca_conc₊Cai * (5 - intra_Ca_conc₊Ca_up)
        intra_Na_conc₊Vi ~ 3.14159265400000010 * 0.05^2 * 2 * (1 - 0.1)
        mem₊RTONF ~ 8314.472 * 310 / 9.64853414999999950e4
        sec_in_curr₊i_si ~ sec_in_curr₊i_siCa + sec_in_curr₊i_siK + sec_in_curr₊i_siNa
        sec_in_curr₊i_siCa ~ 4 * 15 * (mem₊V - 50 ) / (mem₊RTONF * (1 - exp(-1 * (mem₊V - 50 ) * 2 / mem₊RTONF))) * (intra_Ca_conc₊Cai * exp(100  / mem₊RTONF) - 2 * exp(-2 * (mem₊V - 50 ) / mem₊RTONF)) * sec_in_curr_d_gate₊d * sec_in_curr_f_gate₊f * sec_in_curr_f2_gate₊f2
        sec_in_curr₊i_siK ~ 0.01 * 15 * (mem₊V - 50 ) / (mem₊RTONF * (1 - exp(-1 * (mem₊V - 50 ) / mem₊RTONF))) * (intra_K_conc₊Ki * exp(50  / mem₊RTONF) - ext_K_conc₊Kc * exp(-1 * (mem₊V - 50 ) / mem₊RTONF)) * sec_in_curr_d_gate₊d * sec_in_curr_f_gate₊f * sec_in_curr_f2_gate₊f2
        sec_in_curr₊i_siNa ~ 0.01 * 15 * (mem₊V - 50 ) / (mem₊RTONF * (1 - exp(-1 * (mem₊V - 50 ) / mem₊RTONF))) * (intra_Na_conc₊Nai * exp(50  / mem₊RTONF) - 140 * exp(-1 * (mem₊V - 50 ) / mem₊RTONF)) * sec_in_curr_d_gate₊d * sec_in_curr_f_gate₊f * sec_in_curr_f2_gate₊f2
        sec_in_curr_d_gate₊E0_d ~ mem₊V + 24  - 5 
        sec_in_curr_d_gate₊α_d ~ 30  * sec_in_curr_d_gate₊E0_d / (1 - exp(-1 * sec_in_curr_d_gate₊E0_d / 4 ))
        sec_in_curr_d_gate₊β_d ~ 12  * sec_in_curr_d_gate₊E0_d / (exp(sec_in_curr_d_gate₊E0_d / 10 ) - 1)
        sec_in_curr_f2_gate₊β_f2 ~ intra_Ca_conc₊Cai * 5 / 0.001
        sec_in_curr_f_gate₊E0_f ~ mem₊V + 34 
        sec_in_curr_f_gate₊α_f ~ 6.25  * sec_in_curr_f_gate₊E0_f / (exp(sec_in_curr_f_gate₊E0_f / 4 ) - 1)
        sec_in_curr_f_gate₊β_f ~ 50  / (1 + exp(-1 * (mem₊V + 34 ) / 4 ))
        Na_back_curr₊i_Na_b ~ Na_back_curr₊g_Nab * (mem₊V - hyper_act_curr₊E_Na)
        Na_K_pump₊i_p ~ 125 * ext_K_conc₊Kc / (1 + ext_K_conc₊Kc) * intra_Na_conc₊Nai / (40 + intra_Na_conc₊Nai)
        time_dep_K_curr₊i_K ~ time_dep_K_curr_x_gate₊x * 180 * (intra_K_conc₊Ki - ext_K_conc₊Kc * exp(-mem₊V / mem₊RTONF)) / 140
        time_dep_K_curr_x_gate₊α_x ~ 0.5  * exp(0.0826  * (mem₊V + 50 )) / (1 + exp(0.057  * (mem₊V + 50 )))
        time_dep_K_curr_x_gate₊β_x ~ 1.3  * exp(-0.06  * (mem₊V + 20 )) / (1 + exp(-0.04  * (mem₊V + 20 )))
        time_indep_K_curr₊i_K1 ~ time_indep_K_curr₊g_K1 * ext_K_conc₊Kc / (ext_K_conc₊Kc + 210) * (mem₊V - hyper_act_curr₊E_K) / (1 + exp((mem₊V + 10  - hyper_act_curr₊E_K) * 2 / mem₊RTONF))
        trans_out_curr₊i_to ~ trans_out_curr_s_gate₊s * trans_out_curr₊g_to * (0.2 + ext_K_conc₊Kc / (10 + ext_K_conc₊Kc)) * intra_Ca_conc₊Cai / (0.0005 + intra_Ca_conc₊Cai) * (mem₊V + 10 ) / (1 - exp(-0.2  * (mem₊V + 10 ))) * (intra_K_conc₊Ki * exp(0.5 * mem₊V / mem₊RTONF) - ext_K_conc₊Kc * exp(-0.5 * mem₊V / mem₊RTONF))
        trans_out_curr_s_gate₊α_s ~ 0.033  * exp(-mem₊V / 17 )
        trans_out_curr_s_gate₊β_s ~ 33  / (1 + exp(-(mem₊V + 10 ) / 8 ))

        D(ext_K_conc₊Kc) ~ -0.7 * (ext_K_conc₊Kc - 4) + 1e6 * ext_K_conc₊i_mK / (1e15 * 0.00157 * 9.64853414999999950e4)
        D(fast_Na_curr_h_gate₊h) ~ fast_Na_curr_h_gate₊α_h * (1 - fast_Na_curr_h_gate₊h) - fast_Na_curr_h_gate₊β_h * fast_Na_curr_h_gate₊h
        D(fast_Na_curr_m_gate₊m) ~ fast_Na_curr_m_gate₊α_m * (1 - fast_Na_curr_m_gate₊m) - fast_Na_curr_m_gate₊β_m * fast_Na_curr_m_gate₊m
        D(hyper_act_curr_y_gate₊y) ~ hyper_act_curr_y_gate₊α_y * (1 - hyper_act_curr_y_gate₊y) - hyper_act_curr_y_gate₊β_y * hyper_act_curr_y_gate₊y
        D(intra_Ca_conc₊Ca_rel) ~ 1e6 * (intra_Ca_conc₊i_tr - intra_Ca_conc₊i_rel) / (2 * 1e15 * intra_Ca_conc₊V_rel * 9.64853414999999950e4)
        D(intra_Ca_conc₊Ca_up) ~ 1e6 * (intra_Ca_conc₊i_up - intra_Ca_conc₊i_tr) / (2 * 1e15 * intra_Ca_conc₊V_up * 9.64853414999999950e4)
        D(intra_Ca_conc₊Cai) ~ -1e6 * (sec_in_curr₊i_siCa + Ca_back_curr₊i_Ca_b - 2 * Na_Ca_exchanger₊i_NaCa / (3 - 2) - intra_Ca_conc₊i_rel + intra_Ca_conc₊i_up) / (2 * 1e15 * intra_Na_conc₊Vi * 9.64853414999999950e4)
        D(intra_Ca_conc₊p) ~ intra_Ca_conc₊α_p * (1 - intra_Ca_conc₊p) - intra_Ca_conc₊β_p * intra_Ca_conc₊p
        D(intra_K_conc₊Ki) ~ -1e6 * ext_K_conc₊i_mK / (1e15 * intra_Na_conc₊Vi * 9.64853414999999950e4)
        D(intra_Na_conc₊Nai) ~ -1e6 * (fast_Na_curr₊i_Na + Na_back_curr₊i_Na_b + hyper_act_curr₊i_fNa + sec_in_curr₊i_siNa + Na_K_pump₊i_p * 3 + Na_Ca_exchanger₊i_NaCa * 3 / (3 - 2)) / (1e15 * intra_Na_conc₊Vi * 9.64853414999999950e4)
        D(mem₊V) ~ -(hyper_act_curr₊i_f + time_dep_K_curr₊i_K + time_indep_K_curr₊i_K1 + trans_out_curr₊i_to + Na_back_curr₊i_Na_b + Ca_back_curr₊i_Ca_b + Na_K_pump₊i_p + Na_Ca_exchanger₊i_NaCa + fast_Na_curr₊i_Na + sec_in_curr₊i_si) / 0.075
        D(sec_in_curr_d_gate₊d) ~ sec_in_curr_d_gate₊α_d * (1 - sec_in_curr_d_gate₊d) - sec_in_curr_d_gate₊β_d * sec_in_curr_d_gate₊d
        D(sec_in_curr_f2_gate₊f2) ~ 5 - sec_in_curr_f2_gate₊f2 * (5 + sec_in_curr_f2_gate₊β_f2)
        D(sec_in_curr_f_gate₊f) ~ sec_in_curr_f_gate₊α_f * (1 - sec_in_curr_f_gate₊f) - sec_in_curr_f_gate₊β_f * sec_in_curr_f_gate₊f
        D(time_dep_K_curr_x_gate₊x) ~ time_dep_K_curr_x_gate₊α_x * (1 - time_dep_K_curr_x_gate₊x) - time_dep_K_curr_x_gate₊β_x * time_dep_K_curr_x_gate₊x
        D(trans_out_curr_s_gate₊s) ~ trans_out_curr_s_gate₊α_s * (1 - trans_out_curr_s_gate₊s) - trans_out_curr_s_gate₊β_s * trans_out_curr_s_gate₊s
    end
end

# DiFrancesno Noble 1985 model₊ Converted from cellml to myokit, all parameters except conductances were moved to be hardcoded, and the model was converted to work with ModelingToolkit₊
@mtkbuild noble = NOBLE() split=false

ic = [
    noble.mem₊V => -87.,
    noble.hyper_act_curr_y_gate₊y => 0.2,
    noble.time_dep_K_curr_x_gate₊x => 0.01,
    noble.trans_out_curr_s_gate₊s => 1.,
    noble.fast_Na_curr_m_gate₊m => 0.01,
    noble.fast_Na_curr_h_gate₊h => 0.8,
    noble.sec_in_curr_d_gate₊d => 0.005,
    noble.sec_in_curr_f_gate₊f => 1.,
    noble.sec_in_curr_f2_gate₊f2 => 1.,
    noble.intra_Na_conc₊Nai => 8.,
    noble.intra_Ca_conc₊Cai => 5e-5,
    noble.intra_Ca_conc₊Ca_up => 2.,
    noble.intra_Ca_conc₊Ca_rel => 1.,
    noble.intra_Ca_conc₊p => 1.,
    noble.ext_K_conc₊Kc => 4.,
    noble.intra_K_conc₊Ki => 140.,
]

params = [
    noble.fast_Na_curr₊g_Na => 750.
    noble.Ca_back_curr₊g_Cab => 0.02
    noble.hyper_act_curr₊g_f_K => 3.
    noble.hyper_act_curr₊g_f_Na => 3.
    noble.Na_back_curr₊g_Nab => 0.18
    noble.time_indep_K_curr₊g_K1 => 920.
    noble.trans_out_curr₊g_to => 0.28
]

prob = ODEProblem(noble,
ic,
(0.0, 20.0),
params, jac = true, abstol=1e-12, reltol=1e-10, alg=Rodas4P())

odefun = prob.f
F = (u,p) -> odefun(u,p,0)
J = (u,p) -> odefun.jac(u,p,0)

indexof(sym, syms) = findfirst(isequal(sym),syms)
indexofvar(sym) = indexof(sym, unknowns(noble))
indexofparam(sym) = indexof(sym, parameters(noble))
id_gnab = indexofparam(noble.Na_back_curr₊g_Nab)
id_V = indexofvar(noble.mem₊V)
id_conc = indexofvar.([noble.intra_Na_conc₊Nai,
noble.intra_Ca_conc₊Cai,
noble.intra_Ca_conc₊Ca_up,
noble.intra_Ca_conc₊Ca_rel,
noble.intra_Ca_conc₊p,
noble.ext_K_conc₊Kc,
noble.intra_K_conc₊Ki,])

sol = solve(prob)
plot(sol, vars = [noble.mem₊V])


prob = remake(prob; tspan=(0, 1000.))
sol = solve(prob)
concs = (sol[id_conc,:]./sol[id_conc,1])
n = map(x -> norm(x), eachcol(concs))
plot(sol.t, n)
display(title!("Norm of conc for long run time"))

u0 = sol[end]
prob = remake(prob; tspan=(0, 10.), u0=u0)
sol = solve(prob)
concs = (sol[id_conc,:]./sol[id_conc,1])
plot(sol.t, concs')
display(title!("Normalised concentrations at limit cycle"))

# Period looks to be around 1.5
u0_pulse = sol[end]
prob = remake(prob; tspan=(0, 1.5), u0=u0)
sol_pulse = solve(prob)
plot(sol_pulse, vars = [noble.mem₊V])
display(title!("Voltage for 1 pulse"))

bp = BifurcationProblem(F, prob.u0, prob.p, (@lens _[id_gnab]); J=J,
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

# probtrap, ci = BifurcationKit.generate_ci_problem(PeriodicOrbitTrapProblem(M = 150),
# bp, sol_pulse, 1.5)

# opts_br = ContinuationPar(p_min = 0.05, p_max = 0.4, max_steps = 50, tol_stability = 1e-8)
# brpo_fold = continuation(probtrap, ci, PALC(), opts_br;
# 	verbosity = 3, plot = true,
# 	argspo...
# )

probsh, cish = BifurcationKit.generate_ci_problem(ShootingProblem(M = 150),
bp, prob, sol_pulse, 1.5; alg=Rodas4P(), abstol=1e-12, reltol=1e-10)

opts_br = ContinuationPar(p_min = 0.05, p_max = 0.4, max_steps = 50, tol_stability = 1e-8)
brpo_fold = continuation(probsh, cish, PALC(), opts_br;
	verbosity = 3, plot = true,
	argspo...
)

# probcoll, cicoll = BifurcationKit.generate_ci_problem(PeriodicOrbitOCollProblem(30, 4),
# bp, sol_pulse, 1.5)

# opts_br = ContinuationPar(p_min = 0.05, p_max = 0.4, max_steps = 50, tol_stability = 1e-8)
# brpo_fold = continuation(probcoll, cicoll, PALC(), opts_br;
# 	verbosity = 3, plot = true,
# 	argspo...
# )
