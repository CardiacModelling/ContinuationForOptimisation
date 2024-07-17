using Parameters, Plots
using BifurcationKit
using DifferentialEquations
using ConstructionBase
const BK = BifurcationKit

## Set up problem

# vector field
function hh!(dz, z, p, t=0)
	@unpack am1, am2, am3, bm1, bm2, bm3, ah1, ah2, ah3,
    bh1, bh2, gna, Ena, an1, an2, an3, bn1, bn2,
    bn3, gk, Ek, 
	as1, as2, as3, bs1, bs2, bs3, gs, Es, 
	gl, El, Cm = p

	m, h, n, s, V = z

    alpha_m=-am1*(V+am2)/(exp(-(V+am2)/am3)-1)
    beta_m=bm1*exp(-(V+bm2)/bm3)
    alpha_h=ah1*exp(-(V+ah2)/ah3)
    beta_h=1/(exp(-(V+bh1)/bh2)+1)
    alpha_n=-an1*(V+an2)/(exp(-(V+an2)/an3)-1)
    beta_n=bn1*exp((V+bn2)/bn3)
	alpha_s=-as1*(V+as2)/(exp(-(V+as2)/as3)-1)   
	beta_s=bs1*exp((V+bs2)/bs3)                    
    
	dz[1] = alpha_m*(1-m)-beta_m*m
	dz[2] =	alpha_h*(1-h)-beta_h*h
	dz[3] = alpha_n*(1-n)-beta_n*n
	dz[4] = alpha_s*(1-s)-beta_s*s
    dz[5] = -(gna*m^3*h*(V-Ena)+gk*n^4*(V-Ek)+gs*s^4*(V-Es)+gl*(V-El))/Cm  
	dz
end

# parameter values
params = (am1=0.1, am2=50.0, am3=10.0, bm1=4.0, bm2=75.0, bm3=18.0, ah1=0.07, ah2=75.0, ah3=20.0,
bh1=45.0, bh2=10.0, gna=120.0, Ena=40.0, an1=0.01, an2=65.0, an3=10.0, bn1=0.125, bn2=75.0,
bn3=80.0, gk=25.0, Ek=-87.0, 
as1=0.01, as2=65.0, as3=10.0, bs1=0.125, bs2=75.0, bs3=80.0, gs=10.0, Es=-87.0,
gl=0.3, El=-64.387, Cm=1.0)

# initial condition
z0 = [0.05, 0.6, 0.325, 0.3, -75.0]

# Bifurcation Problem
prob = BifurcationProblem(hh!, z0, params, (@lens _.gk);
	record_from_solution = (x, p) -> (V = x[5]),)


## Search along equillibria lines to find hopf bifurcations

# continuation options
opts_br = ContinuationPar(p_min = 2.0, p_max = 80.0,
# parameters to have a smooth continuation curve
dsmin = 0.001, dsmax = 0.05,
)

# continuation of equilibria
br = continuation(prob, PALC(tangent=Bordered()), opts_br; normC = norminf, bothside=true)

# Plot roughly matches MatCont, this doesn't detection all of the special points, although there may be a setting for it
plot(br, plotfold=false, markersize=3, legend=:topleft)
title!("Bifurcation Diagram: gk")
display(ylabel!("V"))

# Bifurcation at gk=13.65
# Adjusting as1 and bs1 to make the convergence slower (~1000 paces to converge)
params = (am1=0.1, am2=50.0, am3=10.0, bm1=4.0, bm2=75.0, bm3=18.0, ah1=0.07, ah2=75.0, ah3=20.0,
bh1=45.0, bh2=10.0, gna=120.0, Ena=40.0, an1=0.01, an2=65.0, an3=10.0, bn1=0.125, bn2=75.0,
bn3=80.0, gk=13.0, Ek=-87.0, 
as1=1e-6, as2=65.0, as3=10.0, bs1=1e-4, bs2=75.0, bs3=80.0, gs=10.0, Es=-87.0,
gl=0.3, El=-64.387, Cm=1.0)

## Plot solution either side of hopf bifurcation

# initial condition - point near the equilibria where the hopf bifurcation was detected
z0 = [0.45, 0.06, 0.52, 0.52, -51.7]

prob_de = ODEProblem(hh!, z0, (0,25000.), params, reltol=1e-8, abstol=1e-8)
sol = solve(prob_de, Rodas5())
plot(sol.t, sol[5,:])
display(title!("Voltage trace"))
plot(sol.t, sol[4,:])
display(title!("s channel"))

# Simulate to get one pulse
prob_de = ODEProblem(hh!, sol[end], (0,20.), params, reltol=1e-8, abstol=1e-8)
sol_pulse = solve(prob_de, Rodas5())
plot(sol_pulse.t, sol_pulse[5,:])
display(title!("Voltage trace: 1 pulse"))
plot(sol_pulse.t, sol_pulse[4,:])
display(title!("s channel: 1 pulse"))

# New problem because parameters changed, also change 
# continuation options
opts_br = ContinuationPar(p_min = 100., p_max = 145.,
)
prob = BifurcationProblem(hh!, sol_pulse[end], params, (@lens _.gna);
	record_from_solution = (x, p) -> (V = x[5]),)

argspo = (record_from_solution = (x, p) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[5,:]),
				min = minimum(xtt[5,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[5,:]; label = "V", k...)
	end)

probtrap, ci = BK.generate_ci_problem(PeriodicOrbitTrapProblem(M = 150),
prob, sol_pulse, 20.)

opts_po_cont = setproperties(opts_br, max_steps = 50, tol_stability = 1e-8)
brpo_fold = continuation(probtrap, ci, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...
)

plot(brpo_fold)
display(title!("Periodic orbit continuation: Max V"))

plot(brpo_fold.param, brpo_fold.period, label="Period")
display(title!("Periodic orbit continuation: Period"))
