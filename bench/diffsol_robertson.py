def robertson_ode_str(ngroups: int):
    u_i = (
        f"(0:{ngroups}): x = 1,\n"
        f"({ngroups}:{2 * ngroups}): y = 0,\n"
        f"({2 * ngroups}:{3 * ngroups}): z = 0,\n"
    )
    code = (
        """
        k1 { 0.04 }
        k2 { 30000000 }
        k3 { 10000 }
        u_i {
        """
        + u_i
        + """
        }
        F_i {
            -k1 * x_i + k3 * y_i * z_i,
            k1 * x_i - k2 * y_i * y_i - k3 * y_i * z_i,
            k2 * y_i * y_i,
        }
        """
    )
    t_final = 1e10
    return code, t_final