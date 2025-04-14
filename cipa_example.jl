using DifferentialEquations
using Parameters, Plots
using LinearAlgebra, NaNMath
using Accessors
using CellMLToolkit

include("cipa.jl")
using .Cipa

prob = Cipa.prob
sol = solve(prob, Tsit5(); Cipa.solversettings(save=true, maxt=1000.0)...)
display(plot(sol, idxs=49))

function convergence_plot(sol, dt=1000)
    # Plot the change in the states across each pulse
    error = []
    for i in dt:dt:sol.t[end]
        push!(error, sum(abs.(sol(i) - sol(i - dt))))
    end
    # Plot error on a log scale
    plot(1:length(error), error, yscale=:log10)
    title!("Convergence plot")
    xlabel!("Pulse count")
    println(error)
    display(ylabel!("Error"))
end

sol = solve(prob, Tsit5(); saveat=1000.0, save_everystep=false, Cipa.solversettings(save=true, maxt=500000.0)...)
convergence_plot(sol)

sol = solve(prob, Tsit5(); Cipa.solversettings(save=false, maxt=2000000.0)...)


@show lcerror(sol[end], [1.0, 1.0, 1.0, 1.0])
@show lcerror(ic, [1.0, 1.0, 1.0, 1.0])
@show lcerror(ic_conv, [1.0, 1.0, 1.0, 1.0])
