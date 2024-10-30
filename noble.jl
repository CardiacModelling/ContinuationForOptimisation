# Define the noble model and check continuation works
using Parameters, Plots
using BifurcationKit
using DifferentialEquations
const BK = BifurcationKit

function noble!(dz, z, p, t=0)
	@unpack g_Na_sf, g_K_sf, g_L_sf = p

	V, m, h, n = z
	
	alpha_m = 100*(-V-48)/(exp((-V-48)/15)-1)
	beta_m = 120*(V+8)/(exp((V+8)/5)-1)
	alpha_h = 170*exp((-V-90)/20)
	beta_h = 1000/(1+exp((-V-42)/10))
	alpha_n = 0.1*(-V-50)/(exp((-V-50)/10)-1)
	beta_n = 2*exp((-V-90)/80)

	g_Na = g_Na_sf*m^3*h*400000
	g_K1 = g_K_sf*1200*exp((-V-90)/50)+15*exp((V+90)/60)
	g_K2 = g_K_sf*1200*n^4

	i_Leak = g_L_sf*75*(V+60)
	i_Na = (g_Na+140)*(V-40)
	i_K = (g_K1+g_K2)*(V+100)

	dz[1] = -(i_Na+i_K+i_Leak)/12
	dz[2] = alpha_m*(1-m)-beta_m*m
	dz[3] = alpha_h*(1-h)-beta_h*h
	dz[4] = alpha_n*(1-n)-beta_n*n

	dz
end

# parameter values
params = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0)

# initial condition
z0 = [-87.0, 0.01, 0.8, 0.01]
# Initial simulation for 20 seconds
prob_de = ODEProblem(noble!, z0, (0.,20.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, saveat=0.001)
plot(sol, idxs=1)
display(title!("Voltage from initial conditions"))

# Continue simulation for 200 seconds
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, tspan=(0.0, 200.0), save_everystep=false, save_start=false)
z0 = sol.u[end]
# Run for a further 20 seconds
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, saveat=0.001, tspan=(0.0, 20.0), u0=z0)
plot(sol, idxs=1)
display(title!("Voltage from converged state"))

# Preconverged initial conditions (200 seconds)
#z0 =  [-84.42650638605386, 0.011194691249055267, 0.9871116097032278, 0.9740803453921694, 0.0030169619183141624, 
#0.999980237264915, 0.021357337599970914, 0.00017938763862391466, 0.9999950652017846, -0.0031415875085243753]

# Try and get continuation to work
prob_de = remake(prob_de, u0=z0, tspan=(0.0, 0.564))
lens = @optic _.g_Na_sf
bp = BifurcationProblem(noble!, z0, params, lens;
	record_from_solution = (x, p; k...) -> (V = x[1]),)

argspo = (record_from_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		return (max = maximum(xtt[1,:]),
				min = minimum(xtt[1,:]),
				period = getperiod(p.prob, x, p.p))
	end,
	plot_solution = (x, p; k...) -> begin
		xtt = get_periodic_orbit(p.prob, x, p.p)
		plot!(xtt.t, xtt[1,:]; label = "V", k...)
	end)

# 1 pulse solution
sol_pulse = DifferentialEquations.solve(prob_de, Tsit5())

opts_br = ContinuationPar(p_min = 0.9, p_max = 1.1, max_steps = 50, tol_stability = 1e-8, ds=0.1, dsmax=0.1, 
detect_bifurcation=0, detect_fold=false, newton_options=NewtonPar(verbose=true))

# Shooting method
bpsh, cish = BK.generate_ci_problem(ShootingProblem(M=1),
bp, prob_de, sol_pulse, 0.564; alg = Tsit5(), abstol=1e-10, reltol=1e-8)

brpo_sh = continuation(bpsh, cish, PALC(), opts_br;
	verbosity = 3, plot = true,  
	argspo...
)
