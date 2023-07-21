import numpy as np

from ml_control.logger import getLogger
from ml_control.systems import solve_homogeneous_system, get_state_from_final_time_adjoint, solve_optimal_control_problem


def greedy(training_parameters, N, T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M,
           tol=1e-4, max_basis_size=45, cg_params={}, return_errors_and_efficiencies=False,
           spatial_norm=lambda x: np.linalg.norm(x)):
    """Runs the greedy algorithm for the given system and training parameters."""
    max_estimated_error = 0.
    if return_errors_and_efficiencies:
        max_true_error = 0.
        max_efficiency = 0.
        optimal_adjoints = []

    phiT_init = np.zeros(N)

    training_data = None

    logger = getLogger("greedy", level="INFO")

    if return_errors_and_efficiencies:
        logger.warn("Computing true errors and efficiencies as well! This might be costly!")

    logger.info("Select first parameter ...")

    for i, mu in enumerate(training_parameters):
        A = parametrized_A(mu)
        B = parametrized_B(mu)
        x0 = parametrized_x0(mu)
        xT = parametrized_xT(mu)
        phiT_mu = np.zeros(N)
        xT_mu = get_state_from_final_time_adjoint(phiT_mu, np.zeros(N), T, nt, A, B, R_chol)[-1]
        assert np.isclose(spatial_norm(xT_mu), 0.)
        x0_T_mu = solve_homogeneous_system(x0, T, nt, A)[-1]

        estimated_error_mu = spatial_norm((phiT_mu - M @ xT_mu) - M @ (x0_T_mu - xT))

        if estimated_error_mu > max_estimated_error:
            max_estimated_error = estimated_error_mu
            index_max = i

        if return_errors_and_efficiencies:
            phi_opt = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, **cg_params)
            optimal_adjoints.append(phi_opt)
            true_error_mu = spatial_norm(phi_opt)
            if true_error_mu > max_true_error:
                max_true_error = true_error_mu
            if estimated_error_mu / true_error_mu > max_efficiency:
                max_efficiency = estimated_error_mu / true_error_mu

    logger.info(f"Determined first parameter with error {max_estimated_error} ...")
    selected_indices = []
    estimated_errors = [max_estimated_error]
    if return_errors_and_efficiencies:
        true_errors = [max_true_error]
        efficiencies = [max_efficiency]

    # Non-orthonormalized reduced basis
    original_selected_phi = []
    # Orthonormalized reduced basis
    selected_phi = []

    logger.info("Starting greedy parameter selection ...")

    k = 0
    while max_estimated_error > tol and k < max_basis_size:
        logger.info(f"Determined next parameter number {index_max} with error {max_estimated_error} ...")
        selected_indices.append(index_max)

        training_data = []

        with logger.block(f"Parameter selection step {k+1}:"):
            j = selected_indices[-1]
            mu = training_parameters[j]

            A = parametrized_A(mu)
            B = parametrized_B(mu)
            x0 = parametrized_x0(mu)
            xT = parametrized_xT(mu)
            logger.info(f"Computing optimal adjoint for selected parameter mu={mu} ...")
            phiT_mu = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, **cg_params)
            original_selected_phi.append(phiT_mu.copy())

            # Orthonormalization of basis by Gram-Schmidt algorithm
            phi_mu_temp = phiT_mu.copy()
            for p in selected_phi:
                phi_mu_temp -= p.dot(phiT_mu) * p
            phiT_mu = phi_mu_temp / np.linalg.norm(phi_mu_temp)
            selected_phi.append(phiT_mu)

            logger.info("Checking errors on training set ...")
            max_estimated_error = 0.
            if return_errors_and_efficiencies:
                max_true_error = 0.
                max_efficiency = 0.

            for i, mu in enumerate(training_parameters):
                if i not in selected_indices:
                    A = parametrized_A(mu)
                    B = parametrized_B(mu)
                    x0 = parametrized_x0(mu)
                    xT = parametrized_xT(mu)
                    x0_T_mu = solve_homogeneous_system(x0, T, nt, A)[-1]

                    mat = np.array([phi
                                    - M @ get_state_from_final_time_adjoint(phi, np.zeros(N), T, nt, A, B, R_chol)[-1]
                                    for phi in selected_phi]).T
                    phi_reduced_coefficients = np.linalg.solve(mat.T @ mat, mat.T @ M @ (x0_T_mu - xT))
                    projection = mat @ phi_reduced_coefficients

                    training_data.append((mu, phi_reduced_coefficients))

                    estimated_error_mu = spatial_norm(projection - M @ (x0_T_mu - xT))

                    if estimated_error_mu > max_estimated_error:
                        max_estimated_error = estimated_error_mu
                        index_max = i

                    if return_errors_and_efficiencies:
                        phi_opt = optimal_adjoints[i]
                        true_error_mu = spatial_norm(phi_opt - np.array(selected_phi).T @ phi_reduced_coefficients)
                        if true_error_mu > max_true_error:
                            max_true_error = true_error_mu
                        if (not abs(true_error_mu) < 1e-12) and (estimated_error_mu / true_error_mu > max_efficiency):
                            max_efficiency = estimated_error_mu / true_error_mu

            logger.info(f"Maximum estimated error on training set: {max_estimated_error}")
            estimated_errors.append(max_estimated_error)
            if return_errors_and_efficiencies:
                true_errors.append(max_true_error)
                efficiencies.append(max_efficiency)
            k += 1

    logger.info("Finished greedy selection procedure ...")

    if return_errors_and_efficiencies:
        return selected_indices, np.array(selected_phi), np.array(original_selected_phi), estimated_errors,\
            true_errors, efficiencies, np.array(optimal_adjoints), training_data
    return selected_indices, np.array(selected_phi), np.array(original_selected_phi), estimated_errors, training_data
