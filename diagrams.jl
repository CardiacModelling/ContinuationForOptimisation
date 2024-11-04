using Plots

plotParams = (linewidth=2., dpi=300, size=(450,300), legend=false)

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
plot!(x, upperLine, color=:black, label="", xlabel="Parameter", ylabel="State", xticks=([p1, p2], ["p₁", "p₂"]), yticks=nothing; plotParams...)

# Continuation curve
plot!([p1, p1], [lowerLine(p1), initialCondition], label="", color=:blue; plotParams...)
plot!(p1:0.01:p2, lowerLine, color=:blue, label="Continuation"; plotParams...)

# ODE curve
plot!([p2, p2], [upperLine(p2), initialCondition], label="", color=:red; plotParams...)

# Initial condition
hline!([initialCondition], label="Initial Condition", color=:pink; plotParams...)

savefig("results/diagrams/possibleProblemsA.pdf")

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
y = 0:0.01:1
xCurve1(y) = @. -curve1*(y-centery1)^2+centerx1
xCurve2(y) = @. curve2*(y-centery2)^2+centerx2

# TODO dotted lines for unstable parts of limit cycle

plot(xCurve1(y), y, xlim=(0,1), ylim=(0,1), color=:black, label="", left_margin=5Plots.mm; plotParams...)
plot!(xCurve2(y), y, color=:black, label="", xticks=([p1, p2], ["p₁", "p₂"]), yticks=nothing; plotParams...)
xlabel!("Parameter")
ylabel!("State")

# Continuation curve
criticalY = centery1+sqrt((centerx1-p1)/curve1)
plot!([p1, p1], [initialCondition, criticalY], label="", color=:blue; plotParams...)
plot!(xCurve1(criticalY:-0.01:0), criticalY:-0.01:0, color=:blue, label="Continuation"; plotParams...)

# ODE curve
plot!([p2, p2], [initialCondition, centery2-sqrt((p2-centerx2)/curve2)], label="", color=:red; plotParams...)

# Initial Condition
hline!([initialCondition], label="Initial Condition", color=:pink; plotParams...)

savefig("results/diagrams/possibleProblemsB.pdf")
