import DifferentialEquations as DE
import ModelingToolkit as MTK


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