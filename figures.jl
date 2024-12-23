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
    paramNames = ["gNa" "gK" "gL" "σ"]

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
    order = [4, 3, 1, 2] # TODO: Order needs fixing so all are visible
    plot(chain[:,order]./pTrueWithNoise[order]', label=paramNames[order'], xticks=([0,20000,40000],["0","2×10⁵","4×10⁵"]),
    xlabel="Iteration", xlim=(0,numSamples); 
    plot_params...)
    vline!([numSamples*0.25+0.5], label="Burn In", color=:red, linewidth=1.5, linestyle=:dot)
    plots = [plots... vline!([numSamples*0.1+0.5], label="Adaption", color=:green, linewidth=1.5, linestyle=:dot, legend=nothing)]

    # Plot posterior histograms
    for i in axes(posterior, 2)
        histogram(posterior[:, i], normalize=:pdf, bins=35, linecolor=:match,
        legend = false; plot_params...)
        plots = [plots... vline!([pTrueWithNoise[i]], color=:black, linewidth=1.5)]
    end
end

l = @layout [a b c]

function plotter(plots, title)
    plot(plots..., layout=l, size=(539,250), dpi=300, link=:all, bottom_margin=2Plots.mm, right_margin=3Plots.mm, yformatter=:none, title=["A" "B" "C"], titlelocation=:left)
    yaxis!(yformatter=x->x, ylabel=title, subplot=1)
    return plot!(legend=:bottomright, subplot=3)
end

plot_ = plotter(plots[1:7:end], "Acceptance Rate")
plot!(legend=:topright, subplot=3)
savefig("results/mcmc/acceptanceRate.pdf")
plot_ = plotter(plots[2:7:end], "Log Likelihood")
savefig("results/mcmc/logLikelihood.pdf")
plot_ = plotter(plots[3:7:end], "Normalized Parameters")
savefig("results/mcmc/convergence.pdf")

function posterior_plotter(plots, xlims)
    plot(plots..., layout=l, size=(539,250), dpi=300, link=:all, yformatter=:none, title=["A" "B" "C"], titlelocation=:left, bottom_margin=2Plots.mm, right_margin=2Plots.mm)
    yaxis!(yformatter=x->x, ylabel="Density", subplot=1)
    return xlims!(xlims...)
end

plot_ = posterior_plotter(plots[4:7:end], (0.98, 1.01))
xlabel!("gNa")
savefig("results/mcmc/posterior_gNa.pdf")
plot_ = posterior_plotter(plots[5:7:end], (0.985, 1.01))
xlabel!("gK")
xticks!([0.99, 1.0, 1.01], ["0.99", "1", "1.01"])
savefig("results/mcmc/posterior_gK.pdf")
plot_ = posterior_plotter(plots[6:7:end], (0.9, 1.05))
xlabel!("gL")
xticks!([0.9, 0.95, 1.0, 1.05], ["0.9", "0.95", "1", "1.05"])
savefig("results/mcmc/posterior_gL.pdf")
plot_ = posterior_plotter(plots[7:7:end], (1.8, 2.4))
xlabel!("σ")
xticks!([1.8, 2.0, 2.2, 2.4], ["1.8", "2", "2.2", "2.4"])
savefig("results/mcmc/posterior_σ.pdf")
