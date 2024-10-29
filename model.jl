# Module to define model, parameters and ICs
module Model
using Parameters
using BifurcationKit

export ode!, params, ic, ic_conv, slow_idx, plot_idx, cont_params

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

function noble_cont!(dz, z, p, t=0)
	@unpack g_Na_sf, g_K_sf, g_L_sf, na_step, k_step, l_step, step = p

	V, m, h, n = z
	
    g_Na_sf = g_Na_sf + na_step*step
    g_K_sf = g_K_sf + k_step*step
    g_L_sf = g_L_sf + l_step*step

	alpha_m = 100*(-V-48)/(exp((-V-48)/15)-1)
	beta_m = 120*(V+8)/(exp((V+8)/5)-1)
	alpha_h = 170*exp((-V-90)/20)
	beta_h = 1000/(1+exp((-V-42)/10))
	alpha_n = 0.1*(-V-50)/(exp((-V-50)/10)-1)
	beta_n = 2*exp((-V-90)/80)

	g_Na = g_Na_sf*m^3*h*400000
	g_K1 = g_K_sf*1200*exp((-V-90)/50)+15*exp((V+90)/60)
	g_K2 = g_K_sf*1200*n^4

	i_Leak = 75*(V+60)
	i_Na = (g_Na+140)*(V-40)
	i_K = (g_K1+g_K2)*(V+100)

	dz[1] = -(i_Na+i_K+i_Leak)/12
	dz[2] = alpha_m*(1-m)-beta_m*m
	dz[3] = alpha_h*(1-h)-beta_h*h
	dz[4] = alpha_n*(1-n)-beta_n*n

	dz
end

# parameter values
params = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf), Tuple{Float64, Float64, Float64}}
params_cont = (g_Na_sf=1.0, g_K_sf=1.0, g_L_sf=1.0, na_step=0.0, k_step=0.0, l_step=0.0, step=0.0)::NamedTuple{(:g_Na_sf, :g_K_sf, :g_L_sf, :na_step, :k_step, :l_step, :step), Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}}

# initial condition
ic = [-87.0, 0.01, 0.8, 0.01]

# Converged initial conditions - 200s at (abs=1e-10, rel=1e-10) tolerances
ic_conv = [ -8.937179276718057,
0.8653023029331617,
0.0030371533410818727,
0.6155744987359527,
] #update at end

slow_idx = 2

plot_idx = 1

end
