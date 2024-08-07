using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using Plots, DifferentialEquations, BifurcationKit
using LinearAlgebra
using BenchmarkTools

@mtkmodel HH begin
    @parameters begin
        am1
        am2
        am3
        bm1
        bm2
        bm3
        ah1
        ah2
        ah3
        bh1
        bh2
        gna
        Ena
        an1
        an2
        an3
        bn1
        bn2
        bn3
        gk
        Ek
        gl
        El
        Cm
    end
    @variables begin
        m(t)
        h(t)
        n(t)
        V(t)
        alpha_m(t)
        beta_m(t)
        alpha_h(t)
        beta_h(t)
        alpha_n(t)
        beta_n(t)
    end
    @equations begin
        alpha_m~-am1*(V+am2)/(exp(-(V+am2)/am3)-1)
        beta_m~bm1*exp(-(V+bm2)/bm3)
        alpha_h~ah1*exp(-(V+ah2)/ah3)
        beta_h~1/(exp(-(V+bh1)/bh2)+1)
        alpha_n~-an1*(V+an2)/(exp(-(V+an2)/an3)-1)
        beta_n~bn1*exp((V+bn2)/bn3)
        
        D(m) ~ alpha_m*(1-m)-beta_m*m
        D(h) ~	alpha_h*(1-h)-beta_h*h
        D(n) ~ alpha_n*(1-n)-beta_n*n
        D(V) ~ -(gna*m^3*h*(V-Ena)+gk*n^4*(V-Ek)+gl*(V-El))/Cm
    end
end

@mtkbuild hh = HH() split=false

ic = [
hh.m => 0.07301116,
hh.h => 0.4970336,
hh.n => 0.34507967,
hh.V => -72.22502,
]

params = [
hh.am1 => 0.1,
hh.am2 => 50.0,
hh.am3 => 10.0,
hh.bm1 => 4.0,
hh.bm2 => 75.0,
hh.bm3 => 18.0,
hh.ah1 => 0.07,
hh.ah2 => 75.0,
hh.ah3 => 20.0,
hh.bh1 => 45.0,
hh.bh2 => 10.0,
hh.gna => 120.0,
hh.Ena => 40.0,
hh.an1 => 0.01,
hh.an2 => 65.0,
hh.an3 => 10.0,
hh.bn1 => 0.125,
hh.bn2 => 75.0,
hh.bn3 => 80.0,
hh.gk => 23.0,
hh.Ek => -87.0,
hh.gl => 0.3,
hh.El => -64.387,
hh.Cm => 1.0,
]

prob = ODEProblem(hh,
ic,
(0.0, 500.0),
params, jac = true, abstol=1e-12, reltol=1e-10)

odefun = prob.f
F = (u,p) -> odefun(u,p,0)
J = (u,p) -> odefun.jac(u,p,0)
Fop = (u,p,t) -> F(u,p)

indexof(sym, syms) = findfirst(isequal(sym),syms)
indexofvar(sym) = indexof(sym, unknowns(hh))
indexofparam(sym) = indexof(sym, parameters(hh))
id_gl = indexofparam(hh.gl)
id_V = indexofvar(hh.V)

u0_pulse = [0.893057403702776, 0.027997147188931967, 0.6424340522523355, -35.9447581061756]
prob = remake(prob; tspan=(0, 25), u0=u0_pulse, abstol=1e-12, reltol=1e-10)
sol_pulse = solve(prob, Rodas5())
plot(sol_pulse, idxs = id_V)
display(title!("Voltage for 1 pulse"))

bp = BifurcationProblem(F, prob.u0, prob.p, (@lens _[id_gl]); J=J,
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
bp, prob, sol_pulse, 24.15; alg=Rodas5(), abstol=1e-12, reltol=1e-10)

opts_br = ContinuationPar(p_min = 0.05, p_max = 0.4, max_steps = 10000, 
    newton_options = NewtonPar(tol=1e-6, verbose=true),
    )
brpo_fold = continuation(probsh, cish, PALC(), opts_br;
    verbosity = 3, plot = true, normC = norminf,
    argspo...
)
