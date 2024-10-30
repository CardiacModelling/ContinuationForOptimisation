module Tools
using Distributions, DifferentialEquations, Plots

function convergence_plot(sol, dt=1000, u=0)
    if u==0
        u = 1:length(sol.u[1])
    end
    println(u)
    # Plot the change in the states across each pulse
    error = []
    for i in dt:dt:sol.t[end]
        push!(error, norm(((sol(i)-sol(i-dt))./scaling)[u]))
    end
    println(error)
    # Plot error on a log scale
    plot(1:length(error), error, yscale=:log10)
    title!("Convergence plot: " * string(u))
    xlabel!("Pulse count")
    display(ylabel!("Error"))
end

# Check all variables are converged automatically
function auto_converge_check(prob::ODEProblem, ic::Vector{Float64}, p::NamedTuple)::Bool
    # Find the average across the first 10s for each state
    sol = solve(prob, Tsit5(), u0=ic, p=p, tspan=(0.0, 10.0), maxiters=1e9)
    avgs = mean(sol.u)
    # Run for further 80s to try and converge closer
    sol = solve(prob, Tsit5(), u0=sol[end], p=p, tspan=(0.0, 80.0), save_everystep=false, save_start=false, maxiters=1e9)
    # Run for further 10s and get the range of each State
    sol = solve(prob, Tsit5(), u0=sol[end], p=p, tspan=(0.0, 10.0), maxiters=1e9)
    # If avgs is inside the range of the final 10s then it is converged
    sol = stack(sol.u)
    return all(minimum(sol, dims=2) .< avgs .< maximum(sol, dims=2))
end

end
