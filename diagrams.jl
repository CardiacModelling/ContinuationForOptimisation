using Plots, LaTeXStrings

plotParams = (linewidth=2., dpi=300, size=(450,300))
l = @layout [a b]
# Panel A
initialCondition = 1.9
p1 = 0.15
p2 = 0.85
ylims = (0, 5)
lowerLine(x) = @. exp(-5*x+0.75)+0.5
upperLine(x) = @. exp(-6*x+3)+1
x = 0:0.01:1
y1 = lowerLine(x)
y2 = upperLine(x)
plot(x, lowerLine, color=:black, label="", left_margin=5Plots.mm, xlim=(0,1), ylim=ylims; plotParams...)
plot!(x, upperLine, color=:black, label="", xlabel="Parameter", ylabel="State", xticks=([p1, p2], [L"p_1", L"p_2"]), yticks=nothing; plotParams...)

# Continuation curve
plot!([p1, p1], [lowerLine(p1), initialCondition], label="", color=:blue; plotParams...)
plot!(p1:0.01:p2, lowerLine, color=:blue, label="Continuation"; plotParams...)

# ODE curve
plot!([p2, p2], [upperLine(p2), initialCondition], label="", color=:red; plotParams...)

# Initial condition
plotA = hline!([initialCondition], label="Initial Condition", color=:pink, legend=false; plotParams...)

# Panel B
initialCondition = 0.53
p1 = 0.15
p2 = 0.85

centerx1 = 0.3
centerx2 = 0.7
centery1 = 0.25
centery2 = 0.75
curve1 = 8
curve2 = 20
xCurve1(y) = @. -curve1*(y-centery1)^2+centerx1
xCurve2(y) = @. curve2*(y-centery2)^2+centerx2

y = 0:0.01:centery1
plot(xCurve1(y), y, xlim=(0,1), ylim=(0,1), linestyle=:dot, color=:black, label="", left_margin=5Plots.mm; plotParams...)
y = centery1:0.01:1
plot(xCurve1(y), y, xlim=(0,1), ylim=(0,1), color=:black, label="", left_margin=5Plots.mm; plotParams...)
y = 0:0.01:centery2
plot!(xCurve2(y), y, color=:black, label="Stable Limit Cycle"; plotParams...)
y = centery2:0.01:1
plot!(xCurve2(y), y, color=:black, label="Unstable Limit Cycle", linestyle=:dot, xticks=([p1, p2], [L"p_1", L"p_2"]), yticks=nothing; plotParams...)
xlabel!("Parameter")
ylabel!("State")

# Continuation curve
criticalY = centery1+sqrt((centerx1-p1)/curve1)
plot!([p1, p1], [initialCondition, criticalY], label="", color=:blue; plotParams...)
y = criticalY:-0.01:centery1
plot!(xCurve1(y), y, color=:blue, label="Continuation Approach"; plotParams...)
y = centery1:-0.01:0
plot!(xCurve1(y), y, color=:blue, linestyle=:dot, label=""; plotParams...)

# ODE curve
plot!([p2, p2], [initialCondition, centery2-sqrt((p2-centerx2)/curve2)], label="Full Approach", color=:red; plotParams...)

# Initial Condition
plotB = hline!([initialCondition], label="Initial Condition", color=:pink; plotParams...)

plot(plotA, plotB, layout=l, size=(900,300), dpi=300, margin=5Plots.mm, left_margin=10Plots.mm)
annotate!([(-0.1, 5, text("A", 16, :black)), (1.13, 5, text("B", 16, :black))])
#savefig("results/diagrams/possibleProblems.pdf")
