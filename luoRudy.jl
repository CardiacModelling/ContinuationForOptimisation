using DynamicalSystems
using DifferentialEquations
using Revise, Parameters, Plots
using BifurcationKit
const BK = BifurcationKit

# CONSTANTS
C_m = 1  #muF/cm2
Esi = 80 #mV per Tran-- in LR had been dependent on Ca_i
Ek = -77 #mV
EK1 = -84  
Ko = 5.4 #mM
ENa = 54.4 #per LR
G_K = 0.282*sqrt(Ko/5.4) #1 per Life/Death SN --  in LR  #mS/cm2
G_Si = 0.07
G_K1 = 0.6047*sqrt(Ko/5.4) #Life/Death added 0.4 factor 

# TIME DEPENDENT POTASSIUM CURRENT
function alpha_w(V)
    return 0.0005*exp(0.083*(V+50))/(1 + exp(0.057*(V+50)))
end

function beta_w(V)
    return 0.0013*exp(-0.06*(V+20))/(1 + exp(-0.04*(V+20)))
end

function xi(V)
    if V <= -100
        return 1
    else
        return 2.837*(exp(0.04*(V+77))-1)/((V+77)*exp(0.04*(V+35)))
    end
end

function I_K(V, w, xi)
    return G_K*w*xi*(V-Ek)
end

# TIME INDEPENDENT POTASSIUM CURRENT
function alpha_K1(V)
    return 1.02/(1 + exp(0.2385*(V-EK1-59.215)))
end

function beta_K1(V)
    return (0.49124*exp(0.08032*(V-EK1+5.476))+exp(0.06175*(V-EK1-594.31)))/(1 + exp(-0.5143*(V-EK1+4.753)))
end

function I_K1(V, K1x)
    return G_K1*K1x*(V-EK1)
end

# PLATEAU POTASSIUM CURRENT
function Kp(V)
    return 1/(1 + exp((7.488-V)/5.98))
end

function I_Kp(V)
    return 0.0183*Kp(V)*(V-EK1) #EK1 
end

# BACKGROUND CURRENT
function I_b(V, Ib_param) 
    return 0.03921*(V + 59.87)*Ib_param # + Ib_param
end

# CALCIUM CURRENT (SLOW INWARD CURRENT)
function alpha_d(V)
    return 0.095*exp(-0.01*(V-5))/(1 + exp(-0.072*(V-5)))
end

function beta_d(V)
    return 0.07*exp(-0.017*(V+44))/(1 + exp(0.05*(V+44)))
end

function alpha_f(V)
    return 0.012*exp(-0.008*(V+28))/(1 + exp(0.15*(V+28)))
end

function beta_f(V)
    return 0.0065*exp(-0.02*(V+30))/(1 + exp(-0.2*(V+30)))
end

function I_si(d, f, V, gsi_param)  # in Tran paper they have a GSI param instead of 0.09
    return 0.09*gsi_param*d*f*(V-Esi)  #in LR, ESI was dependent on Ca_i, Qu fixes it to a constant
end


# SODIUM CURRENT 
function alpha_h(V)
    if V >= -40
        return 0
    else
        return 0.135*exp((80+V)/-6.8)
    end
end

function alpha_j(V)
    if V >= -40
        return 0
    else
        return (-1.2714*10^5*exp(0.2444*V)-3.474*10^-5*exp(-0.04391*V))*(V+37.78)/(1+exp(0.311*(V+79.23)))
    end
end

function beta_h(V)
    if V >= -40
        return 1/(0.13*(1+exp((V+10.66)/-11.1)))
    else
        return 3.56*exp(0.079*V)+3.1*10^5*exp(0.35*V)
    end
end

function beta_j(V)
    if V >= -40
        return 0.3*exp(-2.535*10^-7*V)/(1+exp(-0.1*(V+32)))
    else
        return 0.1212*exp(-0.01052*V)/(1+exp(-0.1378*(V+40.14)))
    end
end

function alpha_m(V)
    return 0.32*(V+47.13)/(1-exp(-0.1*(V+47.13)))
end

function beta_m(V)
    return 0.08*exp(-V/11)
end

function I_Na(m, h, j, V)
    return 23*m^3*h*j*(V-ENa)
end
 
# STIMULUS 
function I_stim(t)
    if t%3000 < 2
        return -30
    else
        return 0
    end
end

# DIFFERENTIAL EQUATIONS
function d_dot(d, V, tau_d_param)
    return (alpha_d(V)*(1-d) - beta_d(V)*d)/tau_d_param
end

function f_dot(f, V, tau_f_param)
    return (alpha_f(V)*(1-f) - beta_f(V)*f)/tau_f_param
end

function w_dot(w, V, tau_w_param)
    return (alpha_w(V)*(1-w) - beta_w(V)*w)/tau_w_param
end

function K1x_dot(K1x, V)
    return alpha_K1(V)*(1-K1x) - beta_K1(V)*K1x
end

function h_dot(h, V)
    return alpha_h(V)*(1-h) - beta_h(V)*h
end

function j_dot(j, V)
    return alpha_j(V)*(1-j) - beta_j(V)*j
end

function m_dot(m, V)
    return alpha_m(V)*(1-m) - beta_m(V)*m
end

function V_dot(d, f, w, K1x, m, h, j, V, gsi_param, Ib_param, t)
    return -1/C_m*(I_si(d,f,V,gsi_param)+ I_K(V, w, xi(V)) + I_b(V, Ib_param) + I_K1(V, K1x) + I_Kp(V) + I_Na(m, h, j, V) + I_stim(t))
end

function LR_EOM!(du, u, p, t)
    @unpack gsi_param, tau_d_param, tau_f_param, tau_w_param, Ib_param = p
    d, f, w, K1x, m, h, j, V = u
    du[1] = d_dot(d, V, tau_d_param)
    du[2] = f_dot(f, V, tau_f_param)
    du[3] = w_dot(w, V, tau_w_param)
    du[4] = K1x_dot(K1x, V)
    du[5] = m_dot(m, V)
    du[6] = h_dot(h, V)
    du[7] = j_dot(j, V)
    du[8] = V_dot(d, f, w, K1x, m, h, j, V, gsi_param, Ib_param, t)
    du
end

#OOP function for ContinuationPar
function LR_EOM(u, p, t = 21000)
    @unpack gsi_param, tau_d_param, tau_f_param, tau_w_param, Ib_param = p
    d, f, w, K1x, m, h, j, V = u
    ddt = d_dot(d, V, tau_d_param)
    dft = f_dot(f, V, tau_f_param)
    dwt = w_dot(w, V, tau_w_param)
    dK1t = K1x_dot(K1x, V)
    dmt = m_dot(m, V)
    dht = h_dot(h, V)
    djt = j_dot(j, V)
    dVt = V_dot(d, f, w, K1x, m, h, j, V, gsi_param, Ib_param, t)
    return SVector(ddt, dft, dwt, dK1t, dmt, dht, djt, dVt, gsi_param, tau_d_param, tau_f_param, tau_w_param, Ib_param)
end

# BIFURCATION DIAGRAM
u0 = [0.7, 0.4, 0.1, 0.1, 0.9, 0.1, 0.1, -15]  # [d, f, w, K1x, m, h, j, V]

param_LR = (gsi_param = 1.1, tau_d_param = 1.0, tau_f_param = 1.0, tau_w_param = 8.0, Ib_param = 1.0)


prob_ode = ODEProblem(LR_EOM!, u0, (0,10000), param_LR)
sol_ode = solve(prob_ode, Tsit5());
plot(sol_ode)

LR_EOM(u, p) = LR_EOM!(similar(u),u,p, 2000)

prob = BifurcationProblem(LR_EOM, sol_ode(2000), param_LR, (@lens _.Ib_param);
    record_from_solution = (x,p) -> (V = x[8]))

opts_br = ContinuationPar(p_min = 1.0, p_max = 1.2, max_steps = 10000,
    newton_options = NewtonPar(tol = 1e-6, max_iterations = 100))

sol = newton(prob, NewtonPar(verbose=true))

br = BifurcationKit.continuation(prob, PALC(), opts_br; normC = norminf, verbosity = 2)

plot(br)
