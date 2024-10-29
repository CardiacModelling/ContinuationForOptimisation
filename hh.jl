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
    bn3, gk, Ek, gl, El, Cm = p

	m, h, n, V = z

    alpha_m=-am1*(V+am2)/(exp(-(V+am2)/am3)-1)
    beta_m=bm1*exp(-(V+bm2)/bm3)
    alpha_h=ah1*exp(-(V+ah2)/ah3)
    beta_h=1/(exp(-(V+bh1)/bh2)+1)
    alpha_n=-an1*(V+an2)/(exp(-(V+an2)/an3)-1)
    beta_n=bn1*exp((V+bn2)/bn3)
    
	dz[1] = alpha_m*(1-m)-beta_m*m
	dz[2] =	alpha_h*(1-h)-beta_h*h
	dz[3] = alpha_n*(1-n)-beta_n*n
    dz[4] = -(gna*m^3*h*(V-Ena)+gk*n^4*(V-Ek)+gl*(V-El))/Cm
	dz
end

# parameter values
params = (am1=0.1, am2=50.0, am3=10.0, bm1=4.0, bm2=75.0, bm3=18.0, ah1=0.07, ah2=75.0, ah3=20.0,
bh1=45.0, bh2=10.0, gna=120.0, Ena=40.0, an1=0.01, an2=65.0, an3=10.0, bn1=0.125, bn2=75.0,
bn3=80.0, gk=36.0, Ek=-87.0, gl=0.3, El=-64.387, Cm=1.0)

# initial condition
z0 = [0.05, 0.6, 0.325, -75.0]

# Bifurcation Problem
prob = BifurcationProblem(hh!, z0, params, (@optic _.gk);
	record_from_solution = (x, p; k...) -> (V = x[4]),)


## Search along equillibria lines to find hopf bifurcations

# continuation options
opts_br = ContinuationPar(p_min = 2.0, p_max = 80.0,
# parameters to have a smooth continuation curve
dsmin = 0.001, dsmax = 0.05,
)

# continuation of equilibria
br = continuation(prob, PALC(tangent=Bordered()), opts_br; normC = norminf, bothside=true)

# Plot roughly matches MatCont, this doesn't detection all of the special points, although there may be a setting for it
plot(br, plotfold=false, markersize=3, dpi=300, size=(450,300), legend=:topright, xlabel="gK", ylabel="V (mV)", title="Bifurcation Diagram: gK")
savefig("bifurcation_diagram.pdf")

## Plot solution either side of hopf bifurcation

# initial condition - point near the equilibria where the hopf bifurcation was detected
z0 = [0.07301116, 0.4970336, 0.34507967, -72.22502]

prob_de = ODEProblem(hh!, z0, (0,200.), params, reltol=1e-8, abstol=1e-8)
sol = DifferentialEquations.solve(prob_de, Rodas5())
plot(sol.t, sol[4,:])
display(title!("Default Parameters: gk 36.0"))

for gk_p in [24., 23.]
	global params = (am1=0.1, am2=50.0, am3=10.0, bm1=4.0, bm2=75.0, bm3=18.0, ah1=0.07, ah2=75.0, ah3=20.0,
	bh1=45.0, bh2=10.0, gna=120.0, Ena=40.0, an1=0.01, an2=65.0, an3=10.0, bn1=0.125, bn2=75.0,
	bn3=80.0, gk=gk_p, Ek=-87.0, gl=0.3, El=-64.387, Cm=1.0)

	global prob_de = ODEProblem(hh!, z0, (0,200.), params, reltol=1e-8, abstol=1e-8)
	global sol = DifferentialEquations.solve(prob_de, Rodas5())
	plot(sol.t, sol[4,:])
	display(title!("gk: $gk_p"))
end

## Continuation of limit cycle
# Simulation a little bit more from updated ICs
prob_de = ODEProblem(hh!, sol[end], (0,25.), params, reltol=1e-8, abstol=1e-8)
sol_pulse = DifferentialEquations.solve(prob_de, Rodas5())
plot(sol_pulse.t, sol_pulse[4,:])
display(title!("One pulse"))

# New problem because parameters changed, also change 
# continuation options
opts_br = ContinuationPar(p_min = 0.2, p_max = 0.5,
# parameters to have a smooth continuation curve
dsmin = 0.001, dsmax = 0.05,
)
prob = BifurcationProblem(hh!, sol_pulse[end], params, (@optic _.gl);
	record_from_solution = (x, p) -> (V = x[4]),)

argspo = (record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[4,:]),
				min = minimum(xtt[4,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[4,:]; label = "V", k...)
	end)

probtrap, ci = BK.generate_ci_problem(PeriodicOrbitTrapProblem(M = 150),
prob, sol_pulse, 25.)

opts_po_cont = setproperties(opts_br, max_steps = 50, tol_stability = 1e-8)
brpo_fold = continuation(probtrap, ci, PALC(), opts_po_cont;
	verbosity = 3, plot = true,
	argspo...
	)

scene = plot(brpo_fold)
