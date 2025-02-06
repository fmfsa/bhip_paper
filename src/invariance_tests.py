import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


def pooling_factor(sample_beta, sample_mu, E):
    """
    Computes the pooling factor for a variable across environments.
    """
    delta = [sample_beta[:, e] - sample_mu for e in range(sample_beta.shape[1])]
    exp_diff = [np.mean(delta[e]) for e in range(E)]
    var_diff = [np.var(delta[e]) for e in range(E)]

    var_exp_diff = np.var(exp_diff)
    exp_var_diff = np.mean(var_diff)

    # pooling factor
    lambda_pool = 1 - var_exp_diff / exp_var_diff
    return max(0, lambda_pool)

def beta_pooling_factor(sample_beta, E):
    """
    Computes the pooling factor for a variable across various environments.

    Args:
        sample_beta: np.ndarray of shape (n_samples, n_environments), values of beta for each sample and environment.
        E: int, number of environments.

    Returns:
        float: Pooling factor in the range [0, 1].
    """
    env_means = np.mean(sample_beta, axis=0)

    sigma2_between = np.var(env_means)
    sigma2_within = np.mean([np.var(sample_beta[:, e]) for e in range(E)])

    lambda_pool = sigma2_between / (sigma2_between + sigma2_within)
    return lambda_pool


def compute_hdi(sample, prob):
    """
    Compute the HDI (highest density interval) of a sample.
    """
    sample = np.asarray(sample)
    hdi_bounds = az.hdi(sample, hdi_prob=prob)
    return hdi_bounds[0], hdi_bounds[1]

def plot_hdi_rope(sample, hdi_low, hdi_upper, rope_low, rope_upper, var_name):
    """
    Plots the HDI vs ROPE for the given sample.
    """
    sns.histplot(sample, kde=True, bins=30, color="blue", label="Posterior Samples")
    plt.axvline(hdi_low, color="red", linestyle="--")
    plt.axvline(hdi_upper, color="red", linestyle="--", label="HDI Bounds")
    plt.axvline(rope_low, color="green", linestyle="--")
    plt.axvline(rope_upper, color="green", linestyle="--", label="ROPE Bounds")
    #plt.title(f"HDI + ROPE for {var_name}")
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    ax = plt.gca()  # Get the current axes
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.show()

def plot_combined_beta(beta_samples, sd_beta, D, E, X_cols):
    """
    Plots the combined HDI + ROPE for all beta predictors across environments.
    """
    for d in range(D):
        predictor_name = X_cols[d] if d < len(X_cols) else f"Predictor_{d}"
        plt.figure(figsize=(10, 6))

        # Define a list of different line styles
        line_styles = ['--', '-.', ':', '-', '--', '-.', ':', '-']

        for e in range(E):
            hdi_low, hdi_upper = compute_hdi(beta_samples[:, e, d], 0.95)
            rope_low, rope_upper = -sd_beta[d, e], sd_beta[d, e]

            # Get the line style for this environment
            line_style = line_styles[e % len(line_styles)]

            sns.histplot(beta_samples[:, e, d], kde=True, bins=30, label=f"Env {e}")
            plt.axvline(hdi_low, color="red", linestyle=line_style)
            plt.axvline(hdi_upper, color="red", linestyle=line_style, label=f"HDI Bounds (Env {e})")
            plt.axvline(rope_low, color="green", linestyle=line_style)
            plt.axvline(rope_upper, color="green", linestyle=line_style, label=f"ROPE Bounds (Env {e})")

        #plt.title(f"Combined HDI + ROPE for Beta: {predictor_name}")
        plt.xlabel("Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend()
        ax = plt.gca()  # Get the current axes
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

def hdi_rope_test(sample, margin, relaxation, plot= False, var_name=""):
    """
    Conducts an HDI + ROPE test on a sample.
    """
    sample = np.asarray(sample)
    hdi_low, hdi_upper = compute_hdi(sample, 1 - relaxation)
    rope_low = -margin
    rope_upper = margin

    print(f"{var_name} HDI: [{hdi_low:.4f}, {hdi_upper:.4f}]")
    print(f"{var_name} ROPE: [{rope_low:.4f}, {rope_upper:.4f}]")
    if plot:
        plot_hdi_rope(sample, hdi_low, hdi_upper, rope_low, rope_upper, var_name)

    if (hdi_low > rope_upper) or (hdi_upper < rope_low):
        return 'Rejected'  # Outside the ROPE
    elif (hdi_low >= rope_low) and (hdi_upper <= rope_upper):
        return 'Accepted'  # Inside the ROPE
    else:
        return 'Undecided'

def prob_mass_outside_rope(sample, rope_low, rope_high):
    """
    Computes the probability mass of the posterior distribution outside the ROPE.

    Parameters:
    - sample: Array of posterior samples for a variable.
    - rope_low: Lower bound of the ROPE.
    - rope_high: Upper bound of the ROPE.

    Returns:
    - largest_out: The larger of the two probabilities outside the ROPE (left or right).
    - left_out: The fraction of samples below the ROPE.
    - right_out: The fraction of samples above the ROPE.
    """
    sample = np.asarray(sample)

    left_out = np.mean(sample < rope_low)
    right_out = np.mean(sample > rope_high)
    largest_out = max(left_out, right_out)

    return largest_out, left_out, right_out


def invariance_tests(mcmc_samples, D, E, X_cols, alpha_beta=0.05, local_rope="two_sd",
                     alpha_mu=0.05, global_rope="sd", p_thres=0.5, pooling_type="normal",
                     printing=True):
    """
    Perform invariance tests on MCMC samples, including posterior probability of invariance
    and pooling factor computations.

    Args:
        pooling_type: 'normal' for `pooling_factor`, 'beta_pooling' for `beta_pooling_factor`.
    """
    beta_samples = np.asarray(mcmc_samples['beta'])
    mu_samples = np.asarray(mcmc_samples['mu'])

    mu_pass = []
    beta_pass = []
    pool_pass = []

    sd_beta = np.std(beta_samples, axis=0).T
    sd_mu = np.std(mu_samples, axis=0)

    for d in range(D):
        predictor_name = X_cols[d] if d < len(X_cols) else f"Predictor_{d}"

        if printing:
            print(f'--------- Predictor {d} ---------')
            print(f'--------- Predictor {predictor_name} ---------')

        # Local ROPE test for each environment
        if local_rope == "two_sd":
            zero_test_beta = [
                hdi_rope_test(beta_samples[:, e, d], 2 * sd_beta[d, e], alpha_beta,
                              plot=printing, var_name=f"beta_{predictor_name}_env_{e}") for e in range(E)
            ]
        elif local_rope == "sd":
            zero_test_beta = [
                hdi_rope_test(beta_samples[:, e, d], sd_beta[d, e], alpha_beta,
                              plot=printing, var_name=f"beta_{predictor_name}_env_{e}") for e in range(E)
            ]
        elif local_rope == "tenth_sd":
            zero_test_beta = [
                hdi_rope_test(beta_samples[:, e, d], 0.1*sd_beta[d, e], alpha_beta,
                              plot=printing, var_name=f"beta_{predictor_name}_env_{e}") for e in range(E)
            ]
        else:
            zero_test_beta = [
                hdi_rope_test(beta_samples[:, e, d], local_rope, alpha_beta,
                              plot=printing, var_name=f"beta_{predictor_name}_env_{e}") for e in range(E)
            ]

        # Global ROPE test for mu
        if global_rope == "two_sd":
            margin_mu = np.mean([2 * sd_beta[d, e] for e in range(E)])
        elif global_rope == "sd":
            margin_mu = sd_mu[d]
        elif global_rope == "tenth_sd":
            margin_mu = sd_mu[d]
        else:
            margin_mu = global_rope
        zero_test_mu = hdi_rope_test(mu_samples[:, d], margin_mu, alpha_mu,
                                     plot=printing, var_name=f"mu_{predictor_name}")

        # Pooling factor selection
        if pooling_type == "normal":
            pooling = pooling_factor(beta_samples[:, :, d], mu_samples[:, d], E)
        elif pooling_type == "beta_pooling":
            pooling = beta_pooling_factor(beta_samples[:, :, d], E)
        else:
            raise ValueError("Invalid pooling_type. Use 'normal' or 'beta_pooling'.")

        if zero_test_mu == "Rejected":
            mu_pass.append(d)
        if all(test == "Rejected" for test in zero_test_beta):
            beta_pass.append(d)
        if pooling >= p_thres:
            pool_pass.append(d)

        if printing:
            print("Local tests:", zero_test_beta)
            print("Global test for mu:", zero_test_mu)
            print(f"Pooling Factor ({pooling_type}): {pooling:.4f}")

    if printing:
        print('CONCLUSION')
        print("Predictors with significant mu (mu_pass):", mu_pass)
        print("Predictors with significant beta in all environments (beta_pass):", beta_pass)
        print("Predictors with high pooling factor (pool_pass):", pool_pass)

    if printing:
        plot_combined_beta(beta_samples, sd_beta, D, E, X_cols)

    return mu_pass, beta_pass, pool_pass
