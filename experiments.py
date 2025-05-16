# ==============================================================================
# Combined Experiment Script for ICP and BHIP Comparison
# ==============================================================================

# --- Imports ---
import os
import json
import time
import math
import random as py_random # For sampling without replacement

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Required external libraries (ensure installed: pip install ...)
import networkx as nx
import sempler
from causalicp import icp
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS

from src.hierarchical_model import (
    nc_hierarchical_model_general
)
from src.invariance_tests import invariance_tests

# import numpyro.distributions as dist # Import if needed by your model
import arviz as az
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# Configuration
# ==============================================================================
np.random.seed(42) # Seed for overall process

# --- Experiment Variations ---
config_node_counts = [4, 5, 6]           # Number of nodes in DAGs
config_sample_sizes = [50, 500, 2000]         # Samples per environment
config_num_envs = [2, 3]             # Total number of environments (1 obs + N-1 int)
num_experiments_per_config = 1000        # Number of random DAGs per configuration

# --- Algorithm Parameters ---
alpha_icp = 0.05      # Significance level for ICP tests
edge_probability = 0.3 # Probability for an edge in the random DAG

# BHIP MCMC Config
num_warmup_bhip = 500
num_samples_bhip = 1000
# BHIP Test Config (Adjust as needed based on results)
bhip_p_thres = 0.85        # Lowered pooling threshold example
bhip_local_rope = "tenth_sd"    # Wider ROPE example
bhip_global_rope = "tenth_sd"   # Wider ROPE example
bhip_pooling_type = "normal" # Potentially more robust pooling

# --- SCM Parameters ---
# Weights for causal effects (where edges exist)
scm_w_bounds = (1, 5) # Example: U(1, 5) -> Stronger signal
# Noise parameters (mean centered, variance low but positive)
scm_noise_mean_bounds = (0, 0)
scm_noise_variance_bounds = (0.1, 0.3)
# Intervention parameters (mean, variance) for interventional envs
scm_intervention_params = (5, 10)

# --- Output Configuration ---
SAVE_DIR = "/mnt/array/fmfsa/bhip_anonymous" # <<< SET YOUR SAVE PATH HERE

# Create the directory if it doesn't exist
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"INFO: Results will be saved to: {SAVE_DIR}")
except Exception as e:
    print(f"ERROR: Could not create save directory '{SAVE_DIR}': {e}")
    SAVE_DIR = "." # Fallback to current directory
    print(f"Warning: Saving results to current directory instead.")

# Plotting Style
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif", "font.size": 10,
    "axes.labelsize": 12, "axes.titlesize": 14, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.figsize": (8, 5), "figure.dpi": 100,
})


# ==============================================================================
# BHIP Model Placeholders -- MUST BE REPLACED BY USER
# ==============================================================================


# --- DAG Generation Function ---
def generate_random_dag_and_check(num_nodes, edge_prob=0.3, max_attempts=100, ensure_connected=True, ensure_target_has_parent=True):
    # Robust definition from previous responses...
    target_node_index = num_nodes - 1
    ensure_connected=False
    for attempt in range(max_attempts):
        A = (np.random.rand(num_nodes, num_nodes) < edge_prob).astype(int); np.fill_diagonal(A, 0)
        if ensure_connected:
            isolated = any(np.sum(A[:, k]) == 0 and np.sum(A[k, :]) == 0 for k in range(num_nodes))
            if isolated: continue
        if ensure_target_has_parent:
            if target_node_index >= 0 and np.sum(A[:, target_node_index]) == 0: continue
        try:
            G = nx.DiGraph(A)
            if nx.is_directed_acyclic_graph(G): return A, True
        except NameError: # networkx fallback
             A_fallback = np.zeros((num_nodes, num_nodes), dtype=int)
             for i in range(num_nodes):
                 for j in range(i + 1, num_nodes):
                     if np.random.rand() < edge_prob: A_fallback[i, j] = 1
             p = np.random.permutation(num_nodes); A_fallback = A_fallback[p][:, p]
             return A_fallback, True
    print(f"Could not generate valid DAG after {max_attempts} attempts.")
    return np.zeros((num_nodes, num_nodes), dtype=int), False

# --- Metrics Calculation Function ---
def calculate_parent_metrics(true_parents, predicted_parents, all_potential_parents):
    # Definition from previous responses (without MCC)...
    TP = len(true_parents.intersection(predicted_parents))
    FP = len(predicted_parents.difference(true_parents))
    FN = len(true_parents.difference(predicted_parents))
    non_parents = all_potential_parents.difference(true_parents)
    TN = len(non_parents.difference(predicted_parents))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics = {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "precision": precision, "recall": recall, "specificity": specificity, "f1_score": f1}
    return metrics

# --- JSON Conversion Helper ---
def convert_sets_to_lists(obj):
    # Definition from previous response...
    if isinstance(obj, set): return sorted(list(obj))
    elif isinstance(obj, dict): return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_sets_to_lists(elem) for elem in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
    elif isinstance(obj, (np.ndarray,)): return obj.tolist()
    elif isinstance(obj, (np.bool_)): return bool(obj)
    return obj


# ==============================================================================
# Main Experiment Execution
# ==============================================================================

all_config_results = {} # Reset results dict

# --- Outer Loops for Configurations ---
for nodes in config_node_counts:
    for samples in config_sample_sizes:
        for num_envs in config_num_envs:
            config_key = (nodes, samples, num_envs)
            print(f"\n{'='*10} Running Config: Nodes={nodes}, Samples={samples}, Envs={num_envs} {'='*10}")
            config_experiment_results = []
            current_num_nodes = nodes
            current_n_samples_per_env = samples
            current_num_envs = num_envs
            current_target_node_index = current_num_nodes - 1
            current_all_potential_parents = set(range(current_num_nodes - 1))

            # --- Inner Loop for Experiments ---
            for i in range(num_experiments_per_config):
                print(f"\n--- Experiment {i+1}/{num_experiments_per_config} (Config: {config_key}) ---")
                start_time_exp = time.time()
                exp_result = {"experiment_id": i+1, "config": config_key}
                try:
                    # 1. Generate DAG
                    A, is_valid_dag = generate_random_dag_and_check(
                        current_num_nodes, edge_prob=edge_probability,
                        ensure_connected=True, ensure_target_has_parent=True # Use constraints
                    )
                    if not is_valid_dag: raise RuntimeError("DAG generation failed")
                    exp_result["dag_A"] = A
                    true_parents = set(np.where(A[:, current_target_node_index] == 1)[0])
                    exp_result["true_parents"] = true_parents
                    print(f"True Parents: {true_parents}")

                    # 2. Create SCM
                    W = A * np.random.uniform(scm_w_bounds[0], scm_w_bounds[1], A.shape)
                    scm = sempler.LGANM(W, scm_noise_mean_bounds, scm_noise_variance_bounds)

                    # 3. Generate Data
                    data_envs_np = []
                    interventions_performed = {}
                    possible_intervention_nodes = list(range(current_num_nodes - 1))
                    py_random.shuffle(possible_intervention_nodes) # Shuffle nodes to pick from
                    nodes_to_intervene = possible_intervention_nodes[:current_num_envs-1] # Select unique nodes

                    for env_idx in range(current_num_envs):
                        if env_idx == 0:
                            data_raw = scm.sample(n=current_n_samples_per_env)
                            interventions_performed[env_idx] = None
                        else:
                            intervention_node = nodes_to_intervene[env_idx-1] # Get unique node for this env
                            intervention = {intervention_node: scm_intervention_params} # Use configured params
                            print(f"  Env {env_idx}: Intervening on node {intervention_node} with params {scm_intervention_params}")
                            interventions_performed[env_idx] = intervention_node
                            data_raw = scm.sample(n=current_n_samples_per_env, do_interventions=intervention)

                        # Data processing
                        if isinstance(data_raw, pd.DataFrame): data_np = data_raw.to_numpy()
                        elif isinstance(data_raw, np.ndarray): data_np = data_raw
                        else: raise TypeError(f"Unexpected data type: {type(data_raw)}")
                        data_envs_np.append(data_np)
                    exp_result["interventions"] = interventions_performed
                    print(f"Generated data for {current_num_envs} environments.")

                    # Combine & Standardize data
                    X_all = np.concatenate([env_data[:, :current_target_node_index] for env_data in data_envs_np], axis=0)
                    Y_all = np.concatenate([env_data[:, current_target_node_index] for env_data in data_envs_np], axis=0)
                    env_indices = np.concatenate([np.full(current_n_samples_per_env, env_idx) for env_idx in range(current_num_envs)], axis=0)
                    N_total = X_all.shape[0]; D_predictors = X_all.shape[1]; E_envs = current_num_envs
                    predictor_cols = [f"X{j}" for j in range(D_predictors)]
                    x_scaler = StandardScaler(); X_all_std = x_scaler.fit_transform(X_all)
                    y_mean, y_std = Y_all.mean(), Y_all.std(); Y_all_std = (Y_all - y_mean) / y_std if y_std > 1e-6 else Y_all

                    # 4. Run ICP
                    print(f"Running ICP (alpha={alpha_icp})...")
                    start_time_icp = time.time()
                    try:
                        result_icp = icp.fit(data=data_envs_np, target=current_target_node_index, alpha=alpha_icp, verbose=False, color=False)
                        if hasattr(result_icp, 'estimate'): icp_accepted_set = set(result_icp.estimate)
                        else: icp_accepted_set = set(result_icp.accepted_sets[-1])
                        icp_metrics = calculate_parent_metrics(true_parents, icp_accepted_set, current_all_potential_parents)
                        exp_result["icp_accepted_set"] = icp_accepted_set; exp_result["icp_metrics"] = icp_metrics
                        exp_result["icp_time"] = time.time() - start_time_icp
                        print(f"ICP Done ({exp_result['icp_time']:.2f}s). Set={icp_accepted_set}, F1={icp_metrics['f1_score']:.3f}")
                    except Exception as icp_e: print(f"Error during ICP: {icp_e}")

                    # 5. Run BHIP
                    print(f"Running BHIP MCMC (Warmup={num_warmup_bhip}, Samples={num_samples_bhip})...")
                    start_time_bhip = time.time()
                    try:
                        X_jax = jnp.array(X_all_std, dtype=jnp.float32); Y_jax = jnp.array(Y_all_std, dtype=jnp.float32); e_jax = jnp.array(env_indices, dtype=jnp.int32)
                        # --- Check if model function is defined ---
                        if 'nc_hierarchical_model_general' not in globals() or not callable(nc_hierarchical_model_general):
                             raise NameError("BHIP model function 'nc_hierarchical_model_general' is not defined correctly.")
                        # --- Run MCMC ---
                        kernel_bhip = NUTS(nc_hierarchical_model_general)
                        mcmc_bhip = MCMC(kernel_bhip, num_warmup=num_warmup_bhip, num_samples=num_samples_bhip, num_chains=1, progress_bar=False)
                        mcmc_bhip.run(random.PRNGKey(i * nodes * samples * num_envs), N=N_total, D=D_predictors, E=E_envs, e=e_jax, X=X_jax, y=Y_jax)
                        posterior_samples = mcmc_bhip.get_samples()

                        # 6. Perform BHIP Invariance Tests
                        # print("Performing BHIP invariance tests...")
                        mu_pass, beta_pass, pool_pass = invariance_tests(
                            posterior_samples, D=D_predictors, E=E_envs, X_cols=predictor_cols, printing=False,
                            p_thres=bhip_p_thres, pooling_type=bhip_pooling_type,
                            local_rope=bhip_local_rope, global_rope=bhip_global_rope
                        )
                        # --- Using intersection of mu_pass and pool_pass as suggested ---
                        bhip_accepted_set = set(mu_pass).intersection(set(pool_pass))
                        # bhip_accepted_set = set(mu_pass).intersection(set(beta_pass)).intersection(set(pool_pass)) # Original

                        bhip_metrics = calculate_parent_metrics(true_parents, bhip_accepted_set, current_all_potential_parents)
                        exp_result["bhip_accepted_set"] = bhip_accepted_set; exp_result["bhip_metrics"] = bhip_metrics
                        exp_result["bhip_time"] = time.time() - start_time_bhip
                        print(f"BHIP Done ({exp_result['bhip_time']:.2f}s). Set={bhip_accepted_set}, F1={bhip_metrics['f1_score']:.3f}")
                        # print(f"BHIP Components: mu={mu_pass}, beta={beta_pass}, pool={pool_pass}") # Optional debug

                    except Exception as bhip_e: print(f"Error during BHIP: {bhip_e}")

                    exp_result["total_exp_time"] = time.time() - start_time_exp
                    config_experiment_results.append(exp_result)

                except Exception as e:
                    print(f"ERROR in Experiment {i+1} (Config {config_key}): {e}")
                    config_experiment_results.append({"experiment_id": i+1, "config": config_key, "error": str(e)})

            all_config_results[config_key] = config_experiment_results


# ==============================================================================
# Aggregation, Summary, Plotting, Saving
# ==============================================================================

# --- Aggregate Results ---
print("\n\n" + "="*20 + " Aggregating Results " + "="*20)
plot_data_list = []
summary_data = []
# (Aggregation logic as defined previously, creating plot_data_list and summary_data)
for config_key, results_list in all_config_results.items():
    nodes, samples, num_envs = config_key
    successful_experiments = [res for res in results_list if "error" not in res]
    valid_icp = [res for res in successful_experiments if "icp_metrics" in res and res["icp_metrics"]]
    valid_bhip = [res for res in successful_experiments if "bhip_metrics" in res and res["bhip_metrics"]]
    # Aggregate ICP
    if valid_icp:
        avg_icp_metrics = {k: np.mean([r['icp_metrics'][k] for r in valid_icp]) for k in valid_icp[0]['icp_metrics']}
        avg_icp_time = np.nanmean([r.get('icp_time', np.nan) for r in valid_icp])
        plot_data_list.append({"num_nodes": nodes, "n_samples": samples, "num_envs": num_envs, "method": "ICP", "time": avg_icp_time, **avg_icp_metrics})
        summary_data.append({"Config": f"N={nodes},S={samples},E={num_envs}", "Method": "ICP", "time": avg_icp_time, **avg_icp_metrics}) # Shorten config string
    # Aggregate BHIP
    if valid_bhip:
        avg_bhip_metrics = {k: np.mean([r['bhip_metrics'][k] for r in valid_bhip]) for k in valid_bhip[0]['bhip_metrics']}
        avg_bhip_time = np.nanmean([r.get('bhip_time', np.nan) for r in valid_bhip])
        plot_data_list.append({"num_nodes": nodes, "n_samples": samples, "num_envs": num_envs, "method": "BHIP", "time": avg_bhip_time, **avg_bhip_metrics})
        summary_data.append({"Config": f"N={nodes},S={samples},E={num_envs}", "Method": "BHIP", "time": avg_bhip_time, **avg_bhip_metrics})

# --- Generate Summary Table ---
print("\n\n" + "="*20 + " Detailed Summary Table " + "="*20)
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    table_cols = ["Config", "Method", "f1_score", "recall", "precision", "specificity", "time", "TP", "FP", "TN", "FN"]
    available_cols = [col for col in table_cols if col in summary_df.columns]; summary_df = summary_df[available_cols]
    float_cols = ["f1_score", "recall", "precision", "specificity", "time"]; int_cols = ["TP", "FP", "TN", "FN"]
    for col in float_cols:
         if col in summary_df.columns: summary_df[col] = summary_df[col].map('{:.3f}'.format)
    for col in int_cols:
         if col in summary_df.columns: summary_df[col] = summary_df[col].map('{:.2f}'.format)
    print(summary_df.to_string(index=False))

    # --- Save Summary Table ---
    summary_table_path = os.path.join(SAVE_DIR, 'summary_metrics_table.csv')
    try:
        raw_summary_df = pd.DataFrame(summary_data) # Use unformatted data for CSV
        raw_summary_df = raw_summary_df[available_cols]
        raw_summary_df.to_csv(summary_table_path, index=False, float_format='%.4f')
        print(f"Saved summary table to: {summary_table_path}")
    except Exception as e: print(f"ERROR saving summary table: {e}")
else: print("No valid results for summary table.")


# --- Aggregate Individual Runs for BoxPlots ---
print("\n\n" + "="*20 + " Aggregating Individual Run Results for Plots " + "="*20)
individual_run_data = []
# (Aggregation logic for individual runs as defined previously)
for config_key, results_list in all_config_results.items():
    nodes, samples, num_envs = config_key
    successful_experiments = [res for res in results_list if "error" not in res]
    for res in successful_experiments:
        config_str = f"N={nodes}, S={samples}, E={num_envs}" # Shorten config string
        if "icp_metrics" in res and res["icp_metrics"]:
            individual_run_data.append({"config_str": config_str, "num_nodes": nodes, "n_samples": samples, "num_envs": num_envs, "method": "ICP", **res['icp_metrics'], "time": res.get('icp_time', np.nan)})
        if "bhip_metrics" in res and res["bhip_metrics"]:
            individual_run_data.append({"config_str": config_str, "num_nodes": nodes, "n_samples": samples, "num_envs": num_envs, "method": "BHIP", **res['bhip_metrics'], "time": res.get('bhip_time', np.nan)})

# --- Save Individual Run Data ---
if individual_run_data:
    plot_df_individual = pd.DataFrame(individual_run_data)
    print(f"Aggregated {len(plot_df_individual)} individual run results points.")
    plot_data_path = os.path.join(SAVE_DIR, 'individual_run_metrics.csv')
    try:
        plot_df_individual.to_csv(plot_data_path, index=False, float_format='%.4f')
        print(f"Saved individual run metrics to: {plot_data_path}")
    except Exception as e: print(f"ERROR saving plotting data: {e}")

    # --- Plotting Section (Using Boxplots) ---
    print("\n" + "="*20 + " Generating Box Plots " + "="*20)
    sns.set_style("whitegrid"); sns.set_context("notebook") # Use 'notebook' context for potentially better default sizes

    metrics_to_plot = ["f1_score", "recall", "precision", "specificity", "time"]
    metric_titles = {"f1_score": "F1 Score", "recall": "Recall", "precision": "Precision", "specificity": "Specificity", "time": "Time (s)"}
    config_order = sorted(plot_df_individual['config_str'].unique(), key=lambda x: (int(x.split('N=')[1].split(',')[0]), int(x.split('S=')[1].split(',')[0]), int(x.split('E=')[1])))

    for metric in metrics_to_plot:
        metric_title = metric_titles[metric]
        print(f"-- Generating combined boxplot for: {metric_title} --")
        try:
            plt.figure(figsize=(max(10, 1.5 * len(config_order)), 6)) # Adjust width based on num configs
            ax = sns.boxplot(data=plot_df_individual, x="config_str", y=metric, hue="method", palette="Set2", order=config_order)
            plt.title(f'Distribution of {metric_title} by Configuration')
            plt.xlabel("Configuration (Nodes, Samples, Environments)")
            plt.ylabel(metric_title); plt.xticks(rotation=30, ha='right')
            # Handle potential log scale for time if needed
            if metric == "time" and plot_df_individual[metric].max() > 10 * plot_df_individual[metric].min(): # Basic check if log scale might help
                 plt.yscale('log')
                 plt.ylabel(f"{metric_title} (log scale)")

            plt.legend(title='Method', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            # Save the figure
            plot_filename = f"boxplot_{metric}_all_configs.png"
            plot_filepath = os.path.join(SAVE_DIR, plot_filename)
            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
            print(f"Saved plot to: {plot_filepath}")
            plt.show()
            plt.close() # Close figure
        except Exception as plot_e: print(f"Could not generate/save boxplot for {metric}: {plot_e}")
    print("Plot generation finished.")
else: print("No valid aggregated data for plotting.")


# --- Save Raw Results Data ---
print("\n\n" + "="*20 + " Saving Raw Experiment Results " + "="*20)
# (Saving logic using convert_sets_to_lists and json.dump as defined previously)
if 'all_config_results' in locals() and isinstance(all_config_results, dict):
    raw_results_path = os.path.join(SAVE_DIR, 'all_config_results.json')
    try:
        results_to_save = convert_sets_to_lists(all_config_results)
        with open(raw_results_path, 'w') as f: json.dump(results_to_save, f, indent=2) # Smaller indent
        print(f"Saved raw results dictionary to: {raw_results_path}")
    except Exception as e: print(f"ERROR saving raw results: {e}")
else: print("Skipping raw results save (all_config_results not found).")

print("\n" + "="*20 + " Script Finished " + "="*20)