using DifferentialEquations, Plots

include("./model.jl")
using .Model

include("./tools.jl")
using .Tools

# Define problem
prob = ODEProblem(Model.ode!, Model.ic, (0.0, 1000.0), Model.params, abstol=1e-10, reltol=1e-8, maxiters=1e7)

# Generate solutions from default initial conditions
sol = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e9)
solFine = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e9, saveat=0.001, tspan=(0.0,1.0), u0=sol[end])
sol_pulse = Tools.aligned_sol(sol[end], prob, 0.53)

# Change initial condition for Vm to 0mV
u0 = copy(Model.ic)
u0[1] = 0

prob = remake(prob, u0=u0)

# Generate solutions for new initial conditions
sol2 = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e9)
sol2Fine = DifferentialEquations.solve(prob, Tsit5(), maxiters=1e9, saveat=0.001, tspan=(0.0,1.0), u0=sol2[end])
sol_pulse2 = Tools.aligned_sol(sol2[end], prob, 0.53)

# Check they converge to the same limit cycle
# 1. Verify the different initial conditions work
plot(sol, label="V(0)=-87", idxs=1)
plot!(sol2, label="V(0)=0", idxs=1)
xlims!(0,1)
display(title!("Start of simulation"))

# 2. Check solution at end looks the same, but will be aligned differently
plot(solFine, label="V(0)=-87", idxs=1)
plot!(sol2Fine, label="V(0)=0", idxs=1)
display(title!("End of simulation t∈(1000,1001)"))

# 3. Check match perfectly after alignment
plot(sol_pulse, label="V(0)=-87", idxs=1)
plot!(sol_pulse2, label="V(0)=0", idxs=1)
display(title!("Aligned solutions"))
