# Add concentrations to the Noble model
using Parameters, Plots
using DifferentialEquations
using Accessors

include("./tools.jl")
using .Tools

# Noble_conc model
function noble_conc!(dz, z, p, t=0)
	# Constants
	nao = 135
	ko = 3.8

	# Parameters
	@unpack g_Na_sf, g_K_sf, g_L_sf, conv_rate = p

    # States
	V, m, h, n, nai, ki = z
	
	# Gating transition rates
	alpha_m = 100*(-V-48)/(exp((-V-48)/15)-1)
	beta_m = 120*(V+8)/(exp((V+8)/5)-1)
	alpha_h = 170*exp((-V-90)/20)
	beta_h = 1000/(1+exp((-V-42)/10))
	alpha_n = 0.1*(-V-50)/(exp((-V-50)/10)-1)
	beta_n = 2*exp((-V-90)/80)

	# Conductances
	g_Na = g_Na_sf*m^3*h*400000
	g_K1 = g_K_sf*1200*exp((-V-90)/50)+15*exp((V+90)/60)
	g_K2 = g_K_sf*1200*n^4

	# Nernst potentials
	R = 8.314
	T = 310
	F = 96.485
	ENa = R*T/F*log(nao/nai)
	EK = R*T/F*log(ko/ki)

	# Currents
	i_Leak = g_L_sf*75*(V+60)
	i_Na = (g_Na+140)*(V-ENa)
	i_K = (g_K1+g_K2)*(V-EK)

	# Calculate intra-cellular concentration targets
	ENa_target = 40
	EK_target = -100
	nai_target = nao*exp(-ENa_target*F/(R*T))
	ki_target = ko*exp(-EK_target*F/(R*T))

	# Differential equations
	dz[1] = -(i_Na+i_K+i_Leak)/12
	dz[2] = alpha_m*(1-m)-beta_m*m
	dz[3] = alpha_h*(1-h)-beta_h*h
	dz[4] = alpha_n*(1-n)-beta_n*n
	dz[5] = conv_rate*(-(i_Na)/(1000*F) - (nai-nai_target)/20.0)
	dz[6] = conv_rate*(-(i_K)/(1000*F) - (ki-ki_target)/20.0)

	dz
end

# parameter values
params = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0, conv_rate=1.0)

# initial condition
# z0 = [-87.0, 0.01, 0.8, 0.01, 30, 160]

# initial condition converged for all sf=1, 1000sec, conv_rate=1.0
z0 = [-68.4831140261365,
0.08798274879055851,
0.53892786879803,
0.44814247723133743,
36.928537846865815,
153.9951030364927]

# perturb the parameters to tune the convergence rate
params = @set params.g_Na_sf=1.1
params = @set params.conv_rate=0.75

# Want to converge in close to 100 seconds
prob_de = ODEProblem(noble_conc!, z0, (0.,200.0), params, reltol=1e-8, abstol=1e-10)
sol = DifferentialEquations.solve(prob_de, Tsit5(), maxiters=1e7, saveat=10.0)

println("Convergence check - 90 seconds: ", Tools.auto_converge_check(prob_de, sol(90), params))
println("Convergence check - 110 seconds: ", Tools.auto_converge_check(prob_de, sol(110), params))

plot(sol, idxs=1)
display(title!("Voltage from converged state"))
plot(sol, idxs=5)
display(title!("Intra-cellular sodium from converged state"))
plot(sol, idxs=6)
display(title!("Intra-cellular potassium from converged state"))
