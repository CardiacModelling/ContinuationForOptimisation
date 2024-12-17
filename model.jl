# Module to define model, parameters and ICs
module Model
using Parameters
using BifurcationKit
using NaNMath

export ode!, params, ic, ic_conv, slow_idx, plot_idx, cont_params

# Noble_conc model
function ode!(dz, z, p, t=0)
	# Constants
	nao = 135
	ko = 3.8

	# Parameters
	@unpack g_Na_sf, g_K_sf, g_L_sf = p

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
	ENa = R*T/F*NaNMath.log(nao/nai)
	EK = R*T/F*NaNMath.log(ko/ki)

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
	dz[5] = 2.5*(-(i_Na)/(1000*F) - (nai-nai_target)/20.0)
	dz[6] = 2.5*(-(i_K+i_Leak)/(1000*F) - (ki-ki_target)/20.0)

	dz
end

function ode_cont!(dz, z, p, t=0)
	# Constants
	nao = 135
	ko = 3.8

	@unpack g_Na_sf, g_K_sf, g_L_sf, na_step, k_step, l_step, step = p
	
    g_Na_sf = g_Na_sf + na_step*step
    g_K_sf = g_K_sf + k_step*step
    g_L_sf = g_L_sf + l_step*step

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
	ENa = R*T/F*NaNMath.log(nao/nai)
	EK = R*T/F*NaNMath.log(ko/ki)

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
	dz[5] = 2.5*(-(i_Na)/(1000*F) - (nai-nai_target)/20.0)
	dz[6] = 2.5*(-(i_K+i_Leak)/(1000*F) - (ki-ki_target)/20.0)

	dz
end

# parameter values
params = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf), Tuple{Float64, Float64, Float64}}
params_cont = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0, na_step=0.0, k_step=0.0, l_step=0.0, step=0.0)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf, :na_step, :k_step, :l_step, :step), Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}}

# initial condition
ic = [-87.0, 0.01, 0.8, 0.01, 30, 160]

# initial condition converged for all sf=1, >10000sec
ic_conv = [-79.08979066519076
0.049845290662513204
0.8083925845419085
0.550265317913508
36.957229146330675
153.78051645130222]

plot_idx = 1

slow_idx = [5,6]

end
