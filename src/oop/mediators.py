# Provide interactions between problems and methods depending on the frame of reference

class FrameMediator:
    pass


def Speed(Equations):
    def calculate_sources(
        gamma,
        D,
        step,
        exact_fluxes,
        cur_state,
        cur_state_der,
        array,
        nodes,
        displacement,
        c_f,
        eps,
        k_for_fric,
    ):
        """
        Calculate the right hand side of the shock-change equation

        Parameters
        ----------
            exact_fluxes : array_like
                Momentum flux at the shock
            array: array_like
                 previous solution for source term (need to take only state at shock)
            cur_state    : 4, array
                Current shock state

        Returns
        -------
            Value of the RHS
        """
        coef_friction = c_f * periodic_friction(
            nodes, displacement, eps, k_for_fric
        )
        source_term_momentum = (
            coef_friction[-3] * array[1, -3] * array[1, -3] / array[0, -3] / 2
        )
        # print(source_term_momentum)
        # The momentum flux gradient on the shock
        rho_a, u_a, p_a, lambda_a = (
            cur_state[0],
            cur_state[1],
            cur_state[2],
            cur_state[3],
        )
        m_flux = exact_fluxes  # HERE SHOULD BE ONLY MOMENTUM FLUX
        m_flux_dx = (
            -12.0 * m_flux[0]
            + 75.0 * m_flux[1]
            - 200.0 * m_flux[2]
            + 300.0 * m_flux[3]
            - 300.0 * m_flux[4]
            + 137.0 * m_flux[5]
        )
        m_flux_dx /= 60.0 * step
        vel_dif = D - u_a
        nom = (
            rho_a
            * vel_dif
            * (
                gamma * (rho_a * u_a * vel_dif - 2.0 * p_a)
                + rho_a * (2.0 * D**2 - 3.0 * D * u_a + u_a**2)
            )
        )
        denom = gamma * (2.0 * p_a + rho_a * vel_dif**2) - rho_a * vel_dif**2
        nom_by_denom_sq = nom / denom**2
        # The derivative of the momentum at the shock \rho*u|_s w.r.t. the shock velocity D
        nom_der = nom / vel_dif + rho_a * vel_dif * (
            gamma * rho_a * u_a + rho_a * (4.0 * D - 3.0 * u_a)
        )
        denom_der = 2.0 * rho_a * (gamma - 1.0) * vel_dif
        dD_dm = 1.0 / (nom_der / denom - denom_der * nom_by_denom_sq)
        # The derivative of the momentum at the shock \rho*u|_s w.r.t. the laboratory coordinate \xi
        dm_dxi = 0
        if cur_state_der[0] >= 1e-8:
            nom_der = nom / rho_a + rho_a * vel_dif * (
                gamma * (u_a * vel_dif) + (2.0 * D**2 - 3.0 * D * u_a + u_a**2)
            )
            denom_der = (gamma - 1.0) * vel_dif**2
            dm_drho = nom_der / denom - denom_der * nom_by_denom_sq
            dm_dxi += dm_drho * cur_state_der[0]
        if cur_state_der[1] >= 1e-8:
            nom_der = -nom / vel_dif + rho_a**2 * vel_dif * (
                (gamma - 3.0) * D - (gamma - 1.0) * 2.0 * u_a
            )
            denom_der = 2.0 * (1.0 - gamma) * rho_a * vel_dif
            dm_du = nom_der / denom - denom_der * nom_by_denom_sq
            dm_dxi += dm_du * cur_state_der[1]
        if cur_state_der[2] >= 1e-8:
            nom_der = -2.0 * rho_a * vel_dif * gamma
            denom_der = 2.0 * gamma
            dm_dp = nom_der / denom - denom_der * nom_by_denom_sq
            dm_dxi += dm_dp * cur_state_der[2]
        return -dD_dm * (m_flux_dx + D * dm_dxi + source_term_momentum)


def Position(Equations):
    def calculate_sources():
        return speed
