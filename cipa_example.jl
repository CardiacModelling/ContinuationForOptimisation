using DifferentialEquations
using Parameters, Plots
using LinearAlgebra, NaNMath
using Accessors
using CellMLToolkit

include("cipa.jl")
using .Cipa

prob = Cipa.prob
sol = solve(prob, Tsit5(); Cipa.solversettings(save=true, maxt=1000.0)...)
display(plot(sol, idxs=Cipa.id_V))

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

sol = solve(prob, Tsit5(); saveat=1000.0, save_everystep=false, Cipa.solversettings(save=true, maxt=500000.0)...) # save=true doesn't set any saving options, so we can set them manually
convergence_plot(sol)

sol = solve(prob, Tsit5(); Cipa.solversettings(save=false, maxt=2000000.0)...)

@show Cipa.lcerror(sol[end], ones(14))
@show Cipa.lcerror(ic, ones(14))
@show Cipa.lcerror(ic_conv, ones(14))
