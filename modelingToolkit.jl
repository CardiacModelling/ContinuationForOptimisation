using Revise, ModelingToolkit, LinearAlgebra
using ModelingToolkit: t_nounits as t, D_nounits as D
using DifferentialEquations, Plots
using BifurcationKit
const BK = BifurcationKit

indexof(sym, syms) = findfirst(isequal(sym),syms)

@mtkmodel NMMODEL begin
    @parameters begin
        U0 = 0.3
        τ = 0.013
        J = 3.07
        E0 = -2.0
        τD = 0.200
        τF = 1.5
        τS = 0.007
        α = 1.5
    end
    @variables begin
        E(t) = 0.238616
        x(t) = 0.982747
        u(t) = 0.367876
        SS0(t)
        SS1(t)
    end
    @equations begin
        SS0 ~ J * u * x * E + E0
        SS1 ~ α * log(1 + exp(SS0 / α))
        D(E) ~ (-E + SS1) / τ
        D(x) ~ (1.0 - x) / τD - u * x * E
        D(u) ~ (U0 - u) / τF +  U0 * (1 - u) * E
    end
end

@mtkbuild NMmodel = NMMODEL()

# get the vector field and jacobian
odeprob = ODEProblem(NMmodel, [], (0.,10.), [], jac = true)
odefun = odeprob.f
F = (u,p) -> odefun(u,p,0)
J = (u,p) -> odefun.jac(u,p,0)

#id_E0 = indexof(NMmodel.E0, parameters(NMmodel))
id_E0 = 7
par_tm = [NMmodel.U0 => 0.3, NMmodel.τ => 0.013, NMmodel.J => 3.07, NMmodel.E0 => -2.0, NMmodel.τD => 0.200, NMmodel.τF => 1.5, NMmodel.τS => 0.007, NMmodel.α => 1.5]
p = ModelingToolkit.varmap_to_vars(par_tm, parameters(NMmodel))
u = ModelingToolkit.varmap_to_vars([NMmodel.E => 0.238616, NMmodel.x => 0.982747, NMmodel.u => 0.367876], unknowns(NMmodel))
# we collect the differentials together in a problem

prob = BifurcationProblem(F, u, odeprob.p[1], (@lens _[id_E0]); J = J,
    record_from_solution = (x, p) -> (E = x[1], x = x[2], u = x[3]))

# continuation options
opts_br = ContinuationPar(p_min = -10.0, p_max = -0.9,
    # parameters to have a smooth result
    ds = 0.04, dsmax = 0.05,
    # this is to detect bifurcation points precisely with bisection
    detect_bifurcation = 3,
    # Optional: bisection options for locating bifurcations
    n_inversion = 8, max_bisection_steps = 25, nev = 3)

# continuation of equilibria
br = continuation(prob, PALC(tangent = Bordered()), opts_br; normC = norminf)

display(plot(br, plotfold=false, markersize=3, legend=:topleft))
