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

function setup_lotka_volterra_ode(ngroups)
    function lotka_volterra!(du, u, p, t)
        a, b, c, d = p
        du[1] = a*u[1] - b*u[1]*u[2]
        du[2] = -c*u[2] + d*u[1]*u[2]
        nothing
    end
    u0 = [1.0, 1.0]
    p = [2.0 / 3.0, 4.0 / 3.0, 1.0, 1.0]
    tspan = (0.0, 10.0)
    prob = DE.ODEProblem(lotka_volterra!, u0, tspan, p)
    return prob, tspan
end



function setup(ngroups, tol, method, problem)
    if problem == "robertson_ode"
        prob, tspan = setup_robertson_ode(ngroups)
    elseif problem == "lotka_volterra_ode"
        prob, tspan = setup_lotka_volterra_ode(ngroups)
    else
        error("Unknown problem: $problem")
    end
    @MTK.mtkcompile sys = MTK.modelingtoolkitize(prob)
    prob = DE.ODEProblem(sys, [], tspan, jac=true, sparse=ngroups >= 20)
    if method == "bdf"
        alg = DE.FBDF()
    elseif method == "kencarp3"
        alg = DE.KenCarp3()
    elseif method == "tr_bdf2"
        alg = DE.TRBDF2()
    elseif method == "tsit5"
        alg = DE.Tsit5()
    else
        error("Unknown method: $method")
    end
    return (prob, alg, tol, tspan)
end

function bench(model)
    (prob, alg, tol, tspan) = model
    sol = DE.solve(prob, alg=alg, reltol = tol, abstol = tol, saveat=tspan[2])
    return sol.u[:, end]
end