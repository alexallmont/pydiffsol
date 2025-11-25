import DifferentialEquations as DE
import ModelingToolkit as MTK
include("diffeq_robertson.jl")
include("diffeq_lokta_volterra.jl")


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