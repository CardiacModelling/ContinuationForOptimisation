using Plots, LaTeXStrings
using BenchmarkTools, BenchmarkPlots, StatsPlots
using CSV, DataFrames

# Simulation timings
t = BenchmarkTools.load("results/simTimings/data.json")[1]
l = @layout [a b]

plot(t["Small"]["ODE"], st=:box, yaxis=:log10, dpi=300, size=(450,300), linecolor=:match, 
markerstrokewidth=0, title="Small Perturbation", whisker_range=0)
plot!(t["Small"]["Cont"], st=:box, yaxis=:log10, legend=:bottomleft, xaxis=nothing, linecolor=:match,
markerstrokewidth=0, ylabel="Time (s)", yformatter=x->x/1e9, ylim=(0.05e9, 1e9), whisker_range=0)
plotA = yaxis!(minorgrid=true)
plot(t["Large"]["ODE"], st=:box, yaxis=:log10, dpi=300, size=(450,300), title="Large Perturbation", 
linecolor=:match, markerstrokewidth=0, whisker_range=0)
plot!(t["Large"]["Cont"], st=:box, yaxis=:log10, legend=nothing, xaxis=nothing, linecolor=:match, markerstrokewidth=0,
ylabel="", yformatter=x->x/1e9, ylim=(0.05e9, 1e9), whisker_range=0)
plotB = yaxis!(minorgrid=true)

plot(plotA, plotB, layout=l, size=(539,200), dpi=300, margin=5Plots.mm, link=:y)
annotate!(-1, 1.75e9, text("A", 12, :black), subplot=1)
annotate!(-1, 1.75e9, text("B", 12, :black), subplot=2)
savefig("results/simTimings/simTimings.pdf")

# MCMC
file_types = ["results/mcmc/cont_", "results/mcmc/trackingODE_", "results/mcmc/standardODE_"]
plots = []
numSamples = 40000
for file_type in file_types
    global plots
    # Read the chain data from the CSV file
    data = CSV.read(file_type*"chain.csv", DataFrame)
    accepts = data.Accept
    lls = data.ℓ
    chain = Matrix([data.gNa data.gK data.gL data.σ])

    # Plot results
    plot_params = (linewidth=2., dpi=300)
    paramNames = ["θ₁" "θ₂" "θ₃" "σ"]

    # Plot acceptance rate
    plot([mean(accepts[max(i-499,1):i]) for i in 1:numSamples], xlabel="Iteration",
    ylim=(0,1), label="Acceptance Rate", xlim = (0,numSamples), xticks=([0,20000,40000],["0","2×10⁵","4×10⁵"]); plot_params...)
    vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
    plots = [plots... vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot, legend=nothing)]

    # Plot log likelihood
    plot(lls, xlabel="Iteration", xlim=(0,numSamples),
    label="ℓ", xticks=([0,20000,40000],["0","2×10⁵","4×10⁵"]); plot_params...)
    vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
    plots = [plots... vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot, legend=nothing)]

    # Remove burn in stage to get posterior distribution
    burnIn = round(Int, numSamples*0.25)
    posterior = chain[burnIn+1:end, :]

    # Plot parameter convergence
    pTrueWithNoise = [1.0, 1.0, 1.0, 2.0]
    order = [4, 3, 1, 2]
    plot(chain[:,order]./pTrueWithNoise[order]', label="", xticks=([0,20000,40000],["0","2×10⁵","4×10⁵"]),
    xlabel="Iteration", xlim=(0,numSamples); 
    plot_params...)
    # Hodge podge of lines in the right order for the legend
    for i in 1:4
        plot!([-1], [-1], label=paramNames[i], color=findfirst(i.==order); plot_params...)
    end
    vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
    ylims!(0.7,1.2)
    plots = [plots... vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot, legend=nothing)]

    # Plot posterior histograms
    p = corrplot(posterior, label=paramNames, size=(539,500), xrot=90, fillcolor=:thermal)
    plot!(p, subplot=16, xformatter=x->x)
    for i in 1:4
        for j in 1:4
            if i != j
                scatter!(p, [pTrueWithNoise[i]], [pTrueWithNoise[j]], subplot=(j-1)*4+i, label="", color=:red, marker=:x)
            end
            if i == j
                vline!(p, [pTrueWithNoise[i]], subplot=(j-1)*4+i, label="", color=:red)
            end
        end
    end

    savefig(file_type*"posterior.pdf")
end

l = @layout [a b c]

function plotter(plots, title)
    plot(plots..., layout=l, size=(539,250), dpi=300, link=:all, bottom_margin=2Plots.mm, right_margin=3Plots.mm, yformatter=:none, title=["A" "B" "C"], titlelocation=:left)
    yaxis!(yformatter=x->x, ylabel=title, subplot=1)
    return plot!(legend=:bottomright, subplot=3)
end

plot_ = plotter(plots[1:3:end], "Acceptance Rate")
plot!(legend=:topright, subplot=3)
savefig("results/mcmc/acceptanceRate.pdf")
plot_ = plotter(plots[2:3:end], "Log Likelihood")
savefig("results/mcmc/logLikelihood.pdf")
plot_ = plotter(plots[3:3:end], "Normalized Parameters")
savefig("results/mcmc/convergence.pdf")
