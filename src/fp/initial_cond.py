def calc_initial(file, L, N, all_nodes, NoG, gamma, Q, act_energy):
    """
    Function for calculating the ZND solution on the mesh of the lenght L
    with N nodes and writing it in the hdf5 file

    Parameters
    ----------
        file      : hdf5 file
            File descriptor to write the initial solution
        L         : double
            Length of the computational domain
        N         : int
            Number of nodes in the mesh
        all_nodes : array_like
            Centers of the cells in the mesh
        NoG       : int
            Number of the ghost points
        Q         : double
            Average heat release to calculate the ZND solution

    Returns
    -------
        rate_const  : double
            Rate of the chemical reaction scaled in such a way that
            the half reaction length (\lambda = 0.5) is equalt to 1
        init_array  : array_like
            Initial ZND profile
    """

    def for_reac_rate(y):
        D_CJ = np.sqrt(gamma + (gamma * gamma - 1.0) * Q / 2.0) + np.sqrt(
            (gamma * gamma - 1.0) * Q / 2.0
        )
        V_lam = (
            gamma
            / D_CJ
            / D_CJ
            * (1.0 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                - (D_CJ * D_CJ - gamma)
                / (1.0 + D_CJ * D_CJ)
                / gamma
                * (np.sqrt(1 - y))
            )
        )
        u_lam = (
            1
            / (gamma + 1.0)
            * (D_CJ * D_CJ - gamma)
            / D_CJ
            * (1 + np.sqrt(1 - y))
        )
        p_lam = (
            (1 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                + (D_CJ * D_CJ - gamma) / (1.0 + D_CJ * D_CJ) * (np.sqrt(1 - y))
            )
        )
        omega = (1 - y) * np.exp(-act_energy / p_lam / V_lam)
        return (u_lam - D_CJ) / omega

    def RHS_lambda(x, y):
        rate_const, _ = integrate.quad(for_reac_rate, 0.0, 0.5)
        rate_const = -rate_const  # ???
        return rate_const / for_reac_rate(y)

    def ZND_init_cond(nodes):
        r = ode(RHS_lambda).set_integrator(
            "zvode", method="bdf"
        )  # using pythonfunction
        r.set_initial_value(0.0, 0.0)
        dx = -(np.abs(nodes[0, 1] - nodes[0, 0])) / 2
        z = []
        lam_x = []
        eps = 1e-6
        cur_lam = 0.0
        node_num = -2
        while (np.abs(cur_lam - 1.0) >= eps) and (node_num >= -nodes.shape[1]):
            z.append(r.t)
            lam_x.append(r.y.real)
            while r.successful() and r.t > nodes[0, node_num]:
                r.integrate(r.t + dx)
            cur_lam = r.y.real
            node_num -= 1
        lam_x = lam_x[::-1]
        lam_x_raw = np.array(lam_x).reshape((1, -1))
        # Here we fill the array for lambda with ones
        # because ODE_integrator does not properly calculate the parts of the domain far from zero
        lam_x = np.hstack(
            (
                np.ones(nodes.shape[1] - lam_x_raw.shape[1]).reshape((1, -1)),
                lam_x_raw,
            )
        )
        D_CJ = np.sqrt(gamma + (gamma * gamma - 1.0) * Q / 2.0) + np.sqrt(
            (gamma * gamma - 1.0) * Q / 2.0
        )
        V_x = (
            gamma
            / D_CJ
            / D_CJ
            * (1 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                - (D_CJ * D_CJ - gamma)
                / (1.0 + D_CJ * D_CJ)
                / gamma
                * (np.sqrt(1 - lam_x))
            )
        )
        u_x = (
            1
            / (gamma + 1.0)
            * (D_CJ * D_CJ - gamma)
            / D_CJ
            * (1 + np.sqrt(1 - lam_x))
        )
        p_x = (
            (1 + D_CJ * D_CJ)
            / (gamma + 1.0)
            * (
                1.0
                + (D_CJ * D_CJ - gamma)
                / (1.0 + D_CJ * D_CJ)
                * (np.sqrt(1 - lam_x))
            )
        )
        # Rankine-Hugoniot conditions for variables at x=0
        (rho_a, u_a, p_a, lam_a) = (1.0, 0.0, 1.0, 0.0)
        c_a = np.sqrt(gamma * p_a / rho_a)
        M_a = D_CJ / c_a
        p_s = (
            2.0 * gamma / (gamma + 1.0) * M_a * M_a
            - (gamma - 1.0) / (gamma + 1.0)
        ) * p_a
        rho_s = (
            (gamma + 1.0) * M_a * M_a / (2.0 + (gamma - 1.0) * M_a * M_a)
        ) * rho_a
        u_s = (2.0 / (gamma + 1.0) * (M_a * M_a - 1.0) / (M_a)) * c_a
        #         self.shock_state = np.array(
        #             (rho_s, u_s, p_s, 0.)).reshape((-1, 1))
        #!!!try to do experimental properties

        # rho_new_tr = np.hstack((np.ones(5)*20 ,np.ones(nodes.shape[1]-5)))
        # u_new_tr = np.ones(nodes.shape[1])
        # p_new_tr = np.hstack((np.ones(5)*200 ,np.ones(nodes.shape[1]-5)))
        # p_new_tr = np.hstack((np.ones(5)))
        # lam_new_tr = np.ones(nodes.shape[1])*0
        ZND_array = np.vstack((1 / V_x, u_x, p_x, lam_x))
        # ZND_array = np.vstack((rho_new_tr, u_new_tr,  p_new_tr, lam_new_tr))
        # print(ZND_array)
        return ZND_array

    # ZND solution behind the shock
    eps_ZND = 1e-12
    init_array_non_positive = ZND_init_cond(
        all_nodes[all_nodes <= eps_ZND].reshape(1, -1)
    )

    # Right ghost points
    if NoG:
        ghost_array = init_array_non_positive[:, -1].reshape(-1, 1)
        for i in range(NoG - 1):
            ghost_array = np.hstack(
                (ghost_array, init_array_non_positive[:, -1].reshape(-1, 1))
            )

        init_array = np.hstack((init_array_non_positive, ghost_array))

    else:
        init_array = init_array_non_positive
    # Write data to hdf5 file
    print()
    initial = file.create_dataset(f"E{act_energy}Q{Q}L{L}N{N}", data=init_array)
    rate_const, _ = integrate.quad(for_reac_rate, 0.0, 0.5)
    rate_const = -rate_const  # ???
    initial.attrs["rate_const"] = rate_const
    return rate_const, init_array

def check_initial():
    with h5py.File("initials.hdf5", "a") as file:
        if f"E{act_energy}Q{Q}" + self.group_name in file:
            print("Init cond is already computed")
            data = file.get(f"E{act_energy}Q{Q}" + self.group_name)
            self.rate_const = data.attrs["rate_const"]
            init_array = data[:]
        else:
            print("Computing init cond")
            self.rate_const, init_array = calc_initial(
                file, L, N, self.nodes, self.NoG, gamma, Q, act_energy
            )
