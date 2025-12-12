import DifferentialEquations as DE
import ModelingToolkit as MTK


function setup_robertson_ode(ngroups)
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
    u0 = vcat(ones(ngroups), zeros(2*ngroups))
    p = [0.04, 3e7, 1e4]
    tspan = (0.0, 1e10)
    prob = DE.ODEProblem(rober!, u0, tspan, p)
    return prob, tspan
end