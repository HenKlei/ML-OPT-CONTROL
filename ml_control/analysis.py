import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

from ml_control.machine_learning_models.gaussian_process_regression import GaussianProcessRegressionReducedModel
from ml_control.greedy_algorithm import greedy
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.machine_learning_models.neural_network_reduced_model import NeuralNetworkReducedModel
from ml_control.reduced_model import ReducedModel
from ml_control.systems import solve_optimal_control_problem, solve_system, get_control_from_final_time_adjoint
from ml_control.visualization import plot_greedy_results, plot_final_time_adjoints, plot_controls, plot_final_time_solutions


def run_greedy_and_analysis(training_parameters, test_parameters_plotting,
                            test_parameters_analysis, N, T, nt, parametrized_A, parametrized_B, parametrized_x0,
                            parametrized_xT, R_chol, M, tol, max_basis_size, cg_params, logger,
                            ml_roms_list=[(NeuralNetworkReducedModel, "DNN-ROM", {}),
                                          (KernelReducedModel, "VKOGA-ROM", {}),
                                          (GaussianProcessRegressionReducedModel, "GPR-ROM", {})],
                            spatial_norm=lambda x: np.linalg.norm(x),
                            temporal_norm=lambda u: np.linalg.norm(u),
                            training_data=None, reduced_basis=None):
    """Runs the greedy algorithm, trains the machine learning surrogates and performs an analysis on a test set."""
    # Offline phase
    restarted_analysis = True
    if training_data is None or reduced_basis is None:
        restarted_analysis = False
        selected_indices, reduced_basis, non_orthonormalized_reduced_basis, \
            estimated_errors, true_errors, efficiencies, optimal_adjoints, training_data = greedy(training_parameters,
                N, T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol, M, tol=tol,
                max_basis_size=max_basis_size, cg_params=cg_params, return_errors_and_efficiencies=True,
                spatial_norm=spatial_norm)

    rom = ReducedModel(reduced_basis, N, T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT,
                       R_chol, M, spatial_norm=spatial_norm)

    ml_roms = []
    labels = []
    for ModelClass, name, training_args in ml_roms_list:
        ml_rom = ModelClass(rom, training_data, T, nt, parametrized_A, parametrized_B, parametrized_x0,
                            parametrized_xT, R_chol, M, spatial_norm=spatial_norm)
        ml_rom.train(**training_args)
        ml_roms.append(ml_rom)
        labels.append(name)

    # Plotting some results
    if not restarted_analysis:
        logger.info("Computing coefficients ...")
        projection_coefficients = []
        roms_coefficients = [[] for _ in range(1 + len(ml_roms_list))]

        phiT_init = np.zeros(N)
        for i, mu in enumerate(training_parameters):
            A = parametrized_A(mu)
            B = parametrized_B(mu)
            x0 = parametrized_x0(mu)
            xT = parametrized_xT(mu)
            phiT_mu = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, **cg_params)

            alpha_mu = reduced_basis.dot(phiT_mu)
            projection_coefficients.append(alpha_mu)

            for j, r in enumerate([rom, *ml_roms]):
                roms_coefficients[j].append(r.solve(mu, return_adjoint_coefficients=True))

        projection_coefficients = np.array(projection_coefficients)
        for j, r in enumerate(roms_coefficients):
            roms_coefficients[j] = np.array(r)

        _, singular_values, _ = np.linalg.svd(optimal_adjoints)

        plot_greedy_results(training_parameters, selected_indices, estimated_errors, true_errors, efficiencies, tol,
                            [projection_coefficients, roms_coefficients[0], *roms_coefficients[1:]],
                            ["Projection of optimal control", "RB-ROM", *labels],
                            reduced_basis, non_orthonormalized_reduced_basis, singular_values)

    # Online phase (plotting)
    with logger.block('Running online phase with plotting of results for '
                      f'{len(test_parameters_plotting)} parameters ...'):
        for i, mu in enumerate(test_parameters_plotting):
            logger.info(f'Results for parameter {mu}:')

            A = parametrized_A(mu)
            B = parametrized_B(mu)
            x0 = parametrized_x0(mu)
            xT = parametrized_xT(mu)

            phiT_init = np.zeros(N)

            tic = time.perf_counter()
            phi_opt = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, **cg_params)
            u_opt = get_control_from_final_time_adjoint(phi_opt, T, nt, A, B, R_chol)
            time_full = time.perf_counter() - tic
            x_opt = solve_system(x0, T, nt, A, B, u_opt)
            print(f"Deviation from target state for full model: {spatial_norm(x_opt[-1] - xT)}")

            us_roms = []
            phis_roms = []
            times_roms = []
            xs_final_roms = []

            for r, name in zip([rom, *ml_roms], ["RB-ROM", *labels]):
                tic = time.perf_counter()
                u, phi = r.solve(mu)
                time_red = time.perf_counter() - tic

                us_roms.append(u)
                phis_roms.append(phi)
                times_roms.append(time_red)

                x = solve_system(x0, T, nt, A, B, u)
                xs_final_roms.append(x[-1])
                print(f"Deviation from target state ({name}): {spatial_norm(x[-1] - xT)}")

            for r, phi, name in zip([rom, *ml_roms], phis_roms, ["RB-ROM", *labels]):
                print(f"Error in final time adjoint ({name}): {spatial_norm(phi - phi_opt)}")
                print(f"Estimated error in final time adjoint ({name}): {r.estimate_error(mu)}")

            for u, name in zip(us_roms, ["RB-ROM", *labels]):
                print(f"Error in control ({name}): {temporal_norm(u - u_opt)}")

            print(f"Runtime full model: {time_full}")
            for t, name in zip(times_roms, ["RB-ROM", *labels]):
                print(f"Runtime ({name}): {t}")

            for t, name in zip(times_roms, ["RB-ROM", *labels]):
                print(f"Speedup ({name}): {time_full / t}")

            print()

            fig, axs = plt.subplots(3)
            plot_final_time_adjoints([phi_opt, *phis_roms],
                                     labels=["Optimal adjoint", "Reduced adjoint",
                                             *[f"{name} reduced adjoint" for name in labels]],
                                     show_plot=False, ax=axs[0])
            axs[0].set_title(f"Final time adjoints")
            axs[0].legend()
            plot_controls([u_opt, *us_roms], T,
                          labels=["Optimal control", "Reduced control",
                                  *[f"{name} reduced control" for name in labels]],
                          show_plot=False, ax=axs[1])
            axs[1].set_title(f"Controls")
            axs[1].legend()
            plot_final_time_solutions([xT, x_opt[-1], *xs_final_roms],
                                      labels=["Target state", "Optimal state", "Reduced state",
                                              *[f"{name} reduced state" for name in labels]],
                                      show_plot=False, ax=axs[2])
            axs[2].set_title(f"Final time states")
            axs[2].legend()
            fig.suptitle(f"Results for parameter {mu}")
            plt.show()

    # Online phase (analysis)
    deviations_from_target_state_opt = []
    deviations_from_target_state_roms = [[] for _ in range(1 + len(ml_roms_list))]
    errors_in_final_time_adjoint_roms = [[] for _ in range(1 + len(ml_roms_list))]
    estimated_errors_in_final_time_adjoint_roms = [[] for _ in range(1 + len(ml_roms_list))]
    errors_in_control_roms = [[] for _ in range(1 + len(ml_roms_list))]
    runtimes_opt = []
    runtimes_roms = [[] for _ in range(1 + len(ml_roms_list))]
    speedups_roms = [[] for _ in range(1 + len(ml_roms_list))]

    with logger.block('Running online phase for analysis of results for '
                      f'{len(test_parameters_analysis)} parameters ...'):
        for i, mu in enumerate(test_parameters_analysis):
            logger.info(f'Parameter number {i} ...')

            A = parametrized_A(mu)
            B = parametrized_B(mu)
            x0 = parametrized_x0(mu)
            xT = parametrized_xT(mu)

            tic = time.perf_counter()
            phi_opt = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, **cg_params)
            u_opt = get_control_from_final_time_adjoint(phi_opt, T, nt, A, B, R_chol)
            time_full = time.perf_counter() - tic
            x_opt = solve_system(x0, T, nt, A, B, u_opt)
            deviations_from_target_state_opt.append(spatial_norm(x_opt[-1] - xT))
            runtimes_opt.append(time_full)

            for i, (r, name) in enumerate(zip([rom, *ml_roms], ["RB-ROM", *labels])):
                tic = time.perf_counter()
                u, phi = r.solve(mu)
                time_red = time.perf_counter() - tic

                x = solve_system(x0, T, nt, A, B, u)

                deviations_from_target_state_roms[i].append(spatial_norm(x[-1] - xT))
                errors_in_final_time_adjoint_roms[i].append(spatial_norm(phi - phi_opt))
                estimated_errors_in_final_time_adjoint_roms[i].append(r.estimate_error(mu))
                errors_in_control_roms[i].append(temporal_norm(u - u_opt))
                runtimes_roms[i].append(time_red)
                speedups_roms[i].append(time_full / time_red)

    with logger.block(f'================== RESULTS FOR {len(test_parameters_analysis)} PARAMETERS =================='):
        with logger.block('Average deviation from target state:'):
            logger.info(f'Full model: {np.average(deviations_from_target_state_opt)}')
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(deviations_from_target_state_roms[i])}')
        with logger.block('Average errors in final time adjoint:'):
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(errors_in_final_time_adjoint_roms[i])}')
        with logger.block('Average errors in control:'):
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(errors_in_control_roms[i])}')
        with logger.block('Average run time:'):
            logger.info(f'Full model: {np.average(runtimes_opt)}')
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(runtimes_roms[i])}')
        with logger.block('Average speedup:'):
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(speedups_roms[i])}')

    fig = plt.figure()
    fig.suptitle('Errors and estimated errors')
    ax = fig.add_subplot(111)
    colors = ["r", "b", "g", "y"]
    for i, (name, c) in enumerate(zip(["RB-ROM", *labels], colors)):
        ax.semilogy(np.arange(len(test_parameters_analysis)), errors_in_final_time_adjoint_roms[i], f"-{c}",
                    label=f"Errors {name}")
        ax.semilogy(np.arange(len(test_parameters_analysis)), estimated_errors_in_final_time_adjoint_roms[i], f"--{c}",
                    label=f"Estimated errors {name}")
    ax.set_xlabel('Test parameter number')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

    fig = plt.figure()
    fig.suptitle('Boxplots of errors')
    ax = fig.add_subplot(111)
    ax.boxplot([*errors_in_final_time_adjoint_roms])
    ax.set_yscale('log')
    ax.set_xticklabels(["RB-ROM", *labels])
    ax.set_xlabel('Reduced model')
    ax.set_ylabel('Error')
    plt.show()

    if not restarted_analysis:
        results_greedy = {'tol': tol,
                          'training_parameters': training_parameters,
                          'selected_indices': selected_indices,
                          'estimated_errors': estimated_errors,
                          'true_errors': true_errors,
                          'efficiencies': efficiencies,
                          'singular_values': singular_values,
                          'projection_coefficients': projection_coefficients,
                          'roms_coefficients': roms_coefficients,
                          'reduced_basis': reduced_basis,
                          'non_orthonormalized_reduced_basis': non_orthonormalized_reduced_basis,
                          'training_data': training_data}
    else:
        results_greedy = {'tol': tol,
                          'training_parameters': training_parameters,
                          'reduced_basis': reduced_basis,
                          'training_data': training_data}
    results_analysis = {'test_parameters': test_parameters_analysis,
                        'deviations_from_target_state_opt': deviations_from_target_state_opt,
                        'deviations_from_target_state_roms': deviations_from_target_state_roms,
                        'errors_in_final_time_adjoint_roms': errors_in_final_time_adjoint_roms,
                        'estimated_errors_in_final_time_adjoint_roms': estimated_errors_in_final_time_adjoint_roms,
                        'errors_in_control_roms': errors_in_control_roms,
                        'runtimes_opt': runtimes_opt,
                        'runtimes_roms': runtimes_roms,
                        'speedups_roms': speedups_roms}
    return results_greedy, results_analysis


def write_results_to_file(results_greedy, results_analysis, filepath_prefix,
                          labels=["RB-ROM", "DNN-ROM", "VKOGA-ROM", "GPR-ROM"]):
    """Writes results of greedy algorithm and online analysis to disc."""
    with open(filepath_prefix + 'reduced_basis.pickle', 'wb') as f:
        pickle.dump(results_greedy['reduced_basis'], f)

    with open(filepath_prefix + 'training_data.pickle', 'wb') as f:
        pickle.dump(results_greedy['training_data'], f)

    if 'selected_indices' in results_greedy:
        with open(filepath_prefix + 'greedy_results.txt', 'w') as f:
            f.write("Greedy step\tEstimated errors\tTrue errors\tEfficiencies\tSelected training parameters\n")
            if results_greedy['training_parameters'].ndim == 1:
                selected_params = np.hstack([np.array([None]),
                                             results_greedy['training_parameters'][results_greedy['selected_indices']]])
            else:
                selected_params = np.vstack([np.array([None] * results_greedy['training_parameters'].shape[1]),
                                             results_greedy['training_parameters'][results_greedy['selected_indices']]])
            for i, (e1, e2, e3, e4) in enumerate(zip(results_greedy['estimated_errors'], results_greedy['true_errors'],
                                                     results_greedy['efficiencies'], selected_params)):
                if i == 0:
                    f.write(f"{i}\t{e1}\t{e2}\t{e3}\t ")
                else:
                    f.write(f"{i}\t{e1}\t{e2}\t{e3}\t{e4}")
                f.write("\n")
    if 'singular_values' in results_greedy:
        with open(filepath_prefix + 'singular_values_optimal_adjoints.txt', 'w') as f:
            for i, s in enumerate(results_greedy['singular_values']):
                f.write(f"{i+1}\t{s}\n")

    with open(filepath_prefix + 'analysis_results_summary.txt', 'w') as f:
        for errs, name in zip(results_analysis['errors_in_final_time_adjoint_roms'], labels):
            f.write(f"Maximum error in adjoint ({name}):\t{np.max(errs)}\n")
        f.write("\n")
        for errs, name in zip(results_analysis['errors_in_final_time_adjoint_roms'], labels):
            f.write(f"Average error in adjoint ({name}):\t{np.average(errs)}\n")
        f.write("\n")
        for errs, name in zip(results_analysis['errors_in_control_roms'], labels):
            f.write(f"Maximum error in control ({name}):\t{np.max(errs)}\n")
        f.write("\n")
        for errs, name in zip(results_analysis['errors_in_control_roms'], labels):
            f.write(f"Average error in control ({name}):\t{np.average(errs)}\n")
        f.write("\n")
        f.write(f"Average runtime (Exact solution):\t{np.average(results_analysis['runtimes_opt'])}\n")
        for ts, name in zip(results_analysis['runtimes_roms'], labels):
            f.write(f"Average runtime ({name}):\t{np.average(ts)}\n")
        f.write("\n")
        for s, name in zip(results_analysis['speedups_roms'], labels):
            f.write(f"Average speedup ({name}):\t{np.average(s)}\n")

    with open(filepath_prefix + 'analysis_results_errors.txt', 'w') as f:
        l = [None] * (2 * len(labels))
        l[::2] = results_analysis['errors_in_final_time_adjoint_roms']
        l[1::2] = results_analysis['estimated_errors_in_final_time_adjoint_roms']
        l.insert(0, results_analysis['test_parameters'])
        for i, (param, e1, e2, e3, e4, e5, e6, e7, e8) in enumerate(zip(*l)):
            f.write(f"{i+1}\t{param}\t{e1}\t{e2}\t{e3}\t{e4}\t{e5}\t{e6}\t{e7}\t{e8}")
            f.write("\n")
