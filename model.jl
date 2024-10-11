# Module to define model, parameters and ICs
module Model
using Parameters
using BifurcationKit

export ode!, params, ic, ic_conv, slow_idx, plot_idx, cont_params
# TODO: Optimise the ode function call, minimise memory allocations

function ode!(dz, z, p, t=0)
    am1, am2, am3, bm1, bm2, bm3, ah1, ah2, ah3, bh1, bh2, Ena, an1, an2, an3, bn1, bn2, bn3, 
    Ek, as1, as2, as3, bs1, bs2, bs3, Es, El, Cm = 0.1, 50.0, 10.0, 4.0, 75.0, 18.0, 0.07, 75.0, 
    20.0, 45.0, 10.0, 40.0, 0.01, 65.0, 10.0, 0.125, 75.0, 80.0, -87.0, 1e-6, 65.0, 10.0, 1e-4, 
    75.0, 80.0, -87.0, -64.387, 1.0

	@unpack gna, gk, gs, gl = p

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

function ode_cont!(dz, z, p, t=0)
    am1, am2, am3, bm1, bm2, bm3, ah1, ah2, ah3, bh1, bh2, Ena, an1, an2, an3, bn1, bn2, bn3, 
    Ek, as1, as2, as3, bs1, bs2, bs3, Es, El, Cm = 0.1, 50.0, 10.0, 4.0, 75.0, 18.0, 0.07, 75.0, 
    20.0, 45.0, 10.0, 40.0, 0.01, 65.0, 10.0, 0.125, 75.0, 80.0, -87.0, 1e-6, 65.0, 10.0, 1e-4, 
    75.0, 80.0, -87.0, -64.387, 1.0

	@unpack gna, gk, gs, gl, gna_step, gk_step, gs_step, gl_step, step = p

    gna = gna + gna_step*step
    gk = gk + gk_step*step
    gs = gs + gs_step*step
    gl = gl + gl_step*step

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
params = (gna=120.0, gk=13.0, gs=10.0, gl=0.3)::NamedTuple{(:gna, :gk, :gs, :gl), Tuple{Float64, Float64, Float64, Float64}}
params_cont = (gna=120.0, gk=13.0, gs=10.0, gl=0.3, gna_step=0.0, gk_step=0.0, gs_step=0.0, gl_step=0.0, step=0.0)::NamedTuple{(:gna, :gk, :gs, :gl, :gna_step, :gk_step, :gs_step, :gl_step, :step), Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64}}

# initial condition
ic = [0.05, 0.6, 0.325, 0.3, -75.0]

# Converged initial conditions - 50,000ms at (abs=1e-10, rel=1e-8) tolerances
ic_conv = [0.13752161502545546,
0.4393685543525413,
0.3638744137880853,
0.12343178944696619,
-64.37702302658744,
]

slow_idx = 4

plot_idx = 5

cont_lens = [(@lens _.gna), (@lens _.gk), (@lens _.gs), (@lens _.gl)]

end
