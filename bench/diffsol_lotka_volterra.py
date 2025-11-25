def lokta_volterra_ode_str():
    code = """
    a { 2.0 / 3.0 }
    b { 4.0 / 3.0 }
    c { 1.0 }
    d { 1.0 }
    u_i {
        x = 1.0,
        y = 1.0,
    }
    F_i {
        a * x - b * x * y,
        -c * y + d * x * y,
    }
    """
    t_final = 10.0
    return code, t_final