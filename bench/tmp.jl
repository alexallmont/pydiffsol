import DifferentialEquations as DE
using SciMLBase

ngroups = 1
function rober!(du, u, p, t)
        k₁, k₂, k₃ = p
        y₁ = @view u[1:ngroups]
        y₂ = @view u[ngroups+1:2*ngroups]
        y₃ = @view u[2*ngroups+1:3*ngroups]
        dy₁ = @view du[1:ngroups]
        dy₂ = @view du[ngroups+1:2*ngroups]
        dy₃ = @view du[2*ngroups+1:3*ngroups]
        dy₁ .= -k₁ .* y₁ .+ k₃ .* y₂ .* y₃
        dy₂ .= k₁ .* y₁ .- k₂ .* y₂ .^2 .- k₃ .* y₂ .* y₃
        dy₃ .= k₂ .* y₂ .^2
        nothing
end

function setup()
        u0 = vcat(ones(ngroups), zeros(2*ngroups))
        p = [0.04, 3e7, 1e4]
        tspan = (0.0, 1e10)
        #prob = DE.ODEProblem{true, SciMLBase.AutoSpecialize}(rober!, u0, tspan, p)
        #prob = DE.ODEProblem{true, SciMLBase.FunctionWrapperSpecialize}(rober!, u0, tspan, p)
        #prob = DE.ODEProblem{true, SciMLBase.FullSpecialize}(rober!, u0, tspan, p)
        prob = DE.ODEProblem{true, SciMLBase.NoSpecialize}(rober!, u0, tspan, p)
        return prob
end

setup1 = @elapsed model = setup()
println("Setup time 1: ", setup1)
setup1 = @elapsed model = setup()
println("Setup time 1 (2nd run): ", setup1)

alg1 = DE.FBDF()
solve11 = @elapsed sol1 = DE.solve(model, alg=alg1)
println("solve time 1 (alg1): ", solve11)
solve12 = @elapsed sol1 = DE.solve(model, alg=alg1)
println("solve time 2 (alg1): ", solve12)
alg2 = DE.TRBDF2()
solve13 = @elapsed sol1 = DE.solve(model, alg=alg2)
println("solve time 3 (alg2): ", solve13)
solve14 = @elapsed sol1 = DE.solve(model, alg=alg2)
println("solve time 4 (alg2): ", solve14)