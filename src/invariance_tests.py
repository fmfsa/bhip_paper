import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import Ellipse


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
    plt.axvline(hdi_low, color="red", linestyle="--", label="HDI Lower Bound")
    plt.axvline(hdi_upper, color="red", linestyle="--", label="HDI Upper Bound")
    plt.axvline(rope_low, color="green", linestyle="--", label="ROPE Lower Bound")
    plt.axvline(rope_upper, color="green", linestyle="--", label="ROPE Upper Bound")
    plt.title(f"HDI + ROPE for {var_name}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_combined_beta(beta_samples, sd_beta, D, E, X_cols):
    """
    Plots the combined HDI + ROPE for all beta predictors across environments.
    """
    for d in range(D):
        predictor_name = X_cols[d] if d < len(X_cols) else f"Predictor_{d}"
        plt.figure(figsize=(10, 6))

        for e in range(E):
            hdi_low, hdi_upper = compute_hdi(beta_samples[:, e, d], 0.95)
            rope_low, rope_upper = -sd_beta[d, e], sd_beta[d, e]

            sns.histplot(beta_samples[:, e, d], kde=True, bins=30, label=f"Env {e}")
            plt.axvline(hdi_low, color="red", linestyle="--", label=f"HDI Lower (Env {e})")
            plt.axvline(hdi_upper, color="red", linestyle="--", label=f"HDI Upper (Env {e})")
            plt.axvline(rope_low, color="green", linestyle="--", label=f"ROPE Lower (Env {e})")
            plt.axvline(rope_upper, color="green", linestyle="--", label=f"ROPE Upper (Env {e})")

        plt.title(f"Combined HDI + ROPE for Beta: {predictor_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
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

def invariance_tests_with_dynamic_rope(
    mcmc_samples, D, E, rope_type="sd", pooling_type="normal",
    printing=True, plotting=False, X_cols=None
):
    """
    Perform invariance tests using HDI, ROPE, and probability mass outside ROPE with flexible margin specification.

    Args:
        mcmc_samples: Dictionary of posterior samples, containing 'beta' and 'mu'.
        D: Number of predictors.
        E: Number of environments.
        rope_type: Type of ROPE margin ('sd', 'two_sd', 'tenth_sd', or a fixed value).
        pooling_type: 'normal' for `pooling_factor`, 'beta_pooling' for `beta_pooling_factor`.
        printing: Whether to print detailed outputs.
        plotting: Whether to generate the heatmap.
        X_cols: List of predictor names to use in the output DataFrame and plot.

    Returns:
        df: DataFrame containing invariance metrics for each predictor.
    """
    beta_samples = np.asarray(mcmc_samples['beta'])
    mu_samples = np.asarray(mcmc_samples['mu'])

    if X_cols is None or len(X_cols) != D:
        X_cols = [f"Predictor_{d}" for d in range(D)]

    results = []

    for d in range(D):
        if printing:
            print(f'--------- Predictor {X_cols[d]} ---------')

        # Determine ROPE margin for beta (local)
        if rope_type in ["sd", "two_sd", "tenth_sd"]:
            beta_sd = np.std(beta_samples[:, :, d], axis=0)
            if rope_type == "sd":
                dynamic_rope_margins = beta_sd
            elif rope_type == "two_sd":
                dynamic_rope_margins = 2 * beta_sd
            elif rope_type == "tenth_sd":
                dynamic_rope_margins = 0.1 * beta_sd
        else:
            dynamic_rope_margins = np.full(E, rope_type)

        local_outcomes = []
        for e in range(E):
            rope_margin = dynamic_rope_margins[e]
            largest_out, left_out, right_out = prob_mass_outside_rope(
                beta_samples[:, e, d], -rope_margin, rope_margin
            )
            local_outcomes.append(largest_out)

            if printing:
                print(f"Env {e}: Largest Out: {largest_out:.4f}, Left: {left_out:.4f}, Right: {right_out:.4f}")

        min_local_outcome = min(local_outcomes)

        # Global outcome for mu
        mu_sd = np.std(mu_samples[:, d])
        if rope_type == "sd":
            global_rope_margin = mu_sd
        elif rope_type == "two_sd":
            global_rope_margin = 2 * mu_sd
        elif rope_type == "tenth_sd":
            global_rope_margin = 0.1 * mu_sd
        else:
            global_rope_margin = rope_type

        global_out, left_out, right_out = prob_mass_outside_rope(
            mu_samples[:, d], -global_rope_margin, global_rope_margin
        )
        if printing:
            print(f"Global Mu: Largest Out: {global_out:.4f}, Left: {left_out:.4f}, Right: {right_out:.4f}")

        # Compute the requested pooling factor
        if pooling_type == "normal":
            pooling = pooling_factor(beta_samples[:, :, d], mu_samples[:, d], E)
        elif pooling_type == "beta_pooling":
            pooling = beta_pooling_factor(beta_samples[:, :, d], E)
        else:
            raise ValueError("Invalid pooling_type. Use 'normal' or 'beta_pooling'.")

        if printing:
            print(f"Pooling Factor ({pooling_type}): {pooling:.4f}")

        results.append({
            "Predictor": X_cols[d],
            "Min Local Outcome": min_local_outcome,
            "Global Outcome": global_out,
            "Pooling Factor": pooling,
        })

    df = pd.DataFrame(results)

    if plotting:
        plot_invariance_heatmap(df)
        plot_2d_invariance_map(df)
        plot_2d_invariance_map_lines(df)

    return df

def bootstrap_data(X, y, e, n_bootstrap):
    """
    Generate bootstrap samples for the data.

    Parameters:
    - X: Covariate matrix (shape: [N, D])
    - y: Target vector (shape: [N])
    - e: Environment indices (shape: [N])
    - n_bootstrap: Number of bootstrap samples

    Returns:
    - List of bootstrap samples [(X_boot, y_boot, e_boot)]
    """
    N = X.shape[0]
    bootstraps = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(N, size=N, replace=True)
        bootstraps.append((X[indices], y[indices], e[indices]))
    return bootstraps


def order_predictors_by_invariance(df, weights=(1/3, 1/3, 1/3)):
    """
    Orders predictors based on a composite metric for causal invariance.

    Parameters:
    - df: DataFrame containing invariance metrics.
    - weights: Tuple of weights for (Min Local Outcome, Global Outcome, Pooling Factor).

    Returns:
    - df_sorted: DataFrame sorted by the composite metric in descending order.
    """
    w1, w2, w3 = weights

    df['Composite Metric'] = (
        w1 * df['Min Local Outcome'] +
        w2 * df['Global Outcome'] +
        w3 * df['Pooling Factor']
    )

    df_sorted = df.sort_values(by='Composite Metric', ascending=False).reset_index(drop=True)

    return df_sorted.drop(columns=["Composite Metric"])


def plot_invariance_heatmap(df):
    """
    Plots a heatmap from the invariance test results DataFrame without normalization.

    Parameters:
    - df: DataFrame with invariance metrics (Predictor as a column).
    """
    df_sorted = order_predictors_by_invariance(df)
    df_plot = df_sorted.set_index("Predictor")

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plot, annot=True, cmap="coolwarm", cbar_kws={"label": "Raw Metric Value"}, vmin=0, vmax=1)
    plt.title("Confidence in Predictor Invariance")
    plt.ylabel("Predictors")
    plt.xlabel("Metrics")
    plt.show()


def plot_2d_invariance_map(df):
    """
    Plots a 2D map of predictors based on local and global confidence with pooling factor as color.

    Parameters:
    - df: DataFrame containing 'Min Local Outcome', 'Global Outcome', and 'Pooling Factor'.
    """
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        df['Min Local Outcome'],
        df['Global Outcome'],
        c=df['Pooling Factor'],
        cmap="viridis",
        s=100,
        edgecolor="black",
        vmin=0,  # Pooling factor lower bound
        vmax=1   # Pooling factor upper bound
    )

    for i, row in df.iterrows():
        plt.text(
            row['Min Local Outcome'], row['Global Outcome'],
            row['Predictor'], fontsize=9, ha='right'
        )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Pooling Factor", fontsize=12)

    plt.title("Predictor Invariance Map", fontsize=16)
    plt.xlabel("Min Local Outcome (Local Confidence)", fontsize=12)
    plt.ylabel("Global Outcome (Global Confidence)", fontsize=12)

    plt.xlim(0, 1.1)  # Enforce x-axis limits
    plt.ylim(0, 1.1)  # Enforce y-axis limits
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()

def plot_2d_invariance_map_lines(df):
    """
    Plots a 2D map of predictors with pooling factor on the y-axis and a line from
    Min Local Outcome to Global Outcome on the x-axis.

    Parameters:
    - df: DataFrame containing 'Min Local Outcome', 'Global Outcome', and 'Pooling Factor'.
    """
    plt.figure(figsize=(10, 8))

    for i, row in df.iterrows():
        # Draw a horizontal line for each predictor
        plt.plot(
            [row['Min Local Outcome'], row['Global Outcome']],
            [row['Pooling Factor'], row['Pooling Factor']],
            label=row['Predictor'], marker='o', linestyle='-', markersize=8
        )
        # Annotate the predictor name at the midpoint of the line
        # midpoint_x = (row['Min Local Outcome'] + row['Global Outcome']) / 2
        # plt.text(
        #     midpoint_x, row['Pooling Factor'],
        #     row['Predictor'], fontsize=9, ha='center', va='bottom'
        # )

    plt.title("Predictor Invariance Map", fontsize=16)
    plt.xlabel("Confidence of Effect (Local to Global)", fontsize=12)
    plt.ylabel("Pooling Factor", fontsize=12)

    #plt.xlim(0, 1.1)  # Enforce x-axis limits
    #plt.ylim(0, 1.1)  # Enforce y-axis limits
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", fontsize=9, frameon=False)
    plt.show()

def plot_predictor_heatmaps(bootstrap_results, X_cols, outcome="global"):
    """
    Generate unified heatmaps for each predictor, either local and global outcome densities.
    """
    n_predictors = len(X_cols)
    n_cols = 3
    n_rows = (n_predictors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), constrained_layout=True)
    axes = axes.flatten()

    for idx, predictor in enumerate(X_cols):
        local_values, global_values, pooling_values = [], [], []
        for df in bootstrap_results:
            row = df.loc[df['Predictor'] == predictor]
            if not row.empty:
                local_values.append(row['Min Local Outcome'].values[0])
                global_values.append(row['Global Outcome'].values[0])
                pooling_values.append(row['Pooling Factor'].values[0])

        outcomes = local_values if outcome == "local" else global_values
        combined_pooling =  pooling_values

        sns.kdeplot(
            x=outcomes,
            y=combined_pooling,
            ax=axes[idx],
            fill=True,
            levels=100,
            cmap="viridis",
            thresh=0,
            clip=((0, 1), (0, 1))
        )
        axes[idx].set_title(f"{predictor}", fontsize=14)
        axes[idx].set_xlabel("Outcome (Local + Global)")
        axes[idx].set_ylabel("Pooling Factor")

    for ax in axes[n_predictors:]:
        ax.axis('off')

    plt.suptitle("Unified Heatmaps of Predictor Metrics", fontsize=18, y=1.02)
    plt.show()
