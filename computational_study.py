import time
import networkx as nx
import numpy as np
import jax.numpy as jnp
from jax import random
from numpyro.infer import MCMC, NUTS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Added for boxplots
from multiprocessing import Pool, cpu_count
import functools
import jax
import os
import argparse

from src.hierarchical_model import (
    nc_hierarchical_model_general
)
from src.invariance_tests import invariance_tests
from causalicp import icp
import sempler

np.random.seed(42) # For reproducibility

# Configure JAX to use single-threaded mode
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

# --- Configuration ---
N_SAMPLES_PER_ENV = 200
NUM_NODES_RANGE = range(3, 21)  # From n=3 to n=15
NUM_REPLICATIONS = 100 # Number of experiments per n_nodes setting
ALPHA_ICP = 0.05
EDGE_PROBABILITY = 0.3
NUM_WARMUP_BHIP = 200  # Reduced for faster runs in study
NUM_SAMPLES_BHIP = 500  # Reduced for faster runs in study
NUM_ENVIRONMENTS = 2 # One observational, one interventional
MAX_PARALLEL_JOBS = 20  # Limit parallel jobs to avoid resource contention

# --- Helper Function: Generate DAG ---
def generate_random_dag_and_check(num_nodes, edge_prob=0.3, max_attempts=100):
    """
    Generates a random adjacency matrix and checks if it's a DAG.
    Requires the 'networkx' library to be installed.
    """
    for _ in range(max_attempts):
        A = (np.random.rand(num_nodes, num_nodes) < edge_prob).astype(int)
        np.fill_diagonal(A, 0)
        G = nx.DiGraph(A)
        if nx.is_directed_acyclic_graph(G):
            return A
    # Fallback: generate upper triangular and permute
    A = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                A[i, j] = 1
    p = np.random.permutation(num_nodes)
    A = A[p][:, p]
    return A

# --- Helper Function: Calculate Metrics ---
def calculate_parent_metrics(true_parents, predicted_parents, all_potential_parents):
    """Calculates TP, FP, TN, FN and derived metrics for parent set identification."""
    TP = len(true_parents.intersection(predicted_parents))
    FP = len(predicted_parents.difference(true_parents))
    FN = len(true_parents.difference(predicted_parents))
    # TN calculation needs to consider only the potential parents, not all nodes in the graph
    # If all_potential_parents is {0, 1, ..., k-1} for a target k
    # and true_parents = {p1, p2}, predicted_parents = {p2, p3}
    # Union = {p1, p2, p3}
    # TN = all_potential_parents - Union
    TN = len(all_potential_parents.difference(true_parents.union(predicted_parents)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision, "recall": recall,
        "specificity": specificity, "f1_score": f1
    }
    return metrics

def run_single_replication(rep_idx):
    """Run a single replication of the study for all node sizes."""
    results_list = []
    print(f"=== Starting Replication {rep_idx + 1}/{NUM_REPLICATIONS} ===")
    
    for n_nodes in NUM_NODES_RANGE:
        print(f"--- Running for num_nodes = {n_nodes} (Replication {rep_idx + 1}) ---")
        
        target_node_index = n_nodes - 1
        # Potential parents are all nodes except the target itself
        all_potential_parents = set(range(n_nodes - 1))
        D_predictors = n_nodes - 1

        study_result = {"replication_id": rep_idx, "num_nodes": n_nodes}

        try:
            # 1. Generate DAG
            A = generate_random_dag_and_check(n_nodes, EDGE_PROBABILITY)
            true_parents = set(np.where(A[:, target_node_index] == 1)[0])
            study_result["true_parents"] = list(true_parents) # for storage
            print(f"  True Parents for target {target_node_index}: {true_parents}")

            # 2. Create SCM
            W = A * np.random.uniform(1, 5, A.shape) # Weights between 1 and 5
            noise_mean_bounds = (0, 0.1) # Tighter noise bounds
            noise_variance_bounds = (0.1, 0.2) # Tighter noise bounds
            scm = sempler.LGANM(W, noise_mean_bounds, noise_variance_bounds)

            # 3. Generate Data (Obs + Int)
            data_envs_np_icp = [] # For ICP format
            
            # Observational Data (Environment 0)
            obs_data = scm.sample(n=N_SAMPLES_PER_ENV)
            data_envs_np_icp.append(obs_data)

            # Interventional Data (Environment 1)
            # Intervene on a random non-target node
            possible_intervention_nodes = list(range(n_nodes - 1)) # Cannot intervene on target for parent discovery
            if not possible_intervention_nodes: # Handles n_nodes=1 or n_nodes=2 edge cases if target is last
                 intervention_node = 0 # Default, though n_nodes starts at 3
            else:
                intervention_node = np.random.choice(possible_intervention_nodes)
            
            # Using a fixed, significant intervention for clarity
            intervention = {intervention_node: (np.random.uniform(3,7), np.random.uniform(0.01, 0.1))} # (mean, variance)
            study_result["intervention_node"] = intervention_node
            int_data = scm.sample(n=N_SAMPLES_PER_ENV, do_interventions=intervention)
            data_envs_np_icp.append(int_data)

            # Prepare data for BHIP (concatenated)
            # X_all should contain only predictor columns for the target
            X_all_list = []
            Y_all_list = []
            env_indices_list = []

            for env_idx, env_data in enumerate(data_envs_np_icp):
                X_env = np.delete(env_data, target_node_index, axis=1) # All columns except target
                Y_env = env_data[:, target_node_index]                 # Target column
                
                X_all_list.append(X_env)
                Y_all_list.append(Y_env)
                env_indices_list.append(np.full(N_SAMPLES_PER_ENV, env_idx))

            X_all = np.concatenate(X_all_list, axis=0)
            Y_all = np.concatenate(Y_all_list, axis=0)
            env_indices = np.concatenate(env_indices_list, axis=0)

            N_total = X_all.shape[0]
            # D_predictors already defined
            E_envs = NUM_ENVIRONMENTS # Should be 2
            predictor_cols_bhip = [f"X{j}" for j in range(D_predictors)]

            # 4. Run ICP
            print(f"  Running ICP (alpha={ALPHA_ICP})...")
            icp_start_time = time.perf_counter()
            # ICP expects list of (X,y) tuples or list of nd.arrays and target index
            # Format for causalicp.fit: data=[env1_data, env2_data, ...], target=target_idx
            result_icp = icp.fit(data=data_envs_np_icp, target=target_node_index, alpha=ALPHA_ICP, verbose=False, color=False)
            icp_end_time = time.perf_counter()
            
            icp_accepted_set = set(result_icp.estimate)
            icp_metrics = calculate_parent_metrics(true_parents, icp_accepted_set, all_potential_parents)
            study_result["icp_accepted_set"] = list(icp_accepted_set)
            study_result["icp_f1"] = icp_metrics["f1_score"]
            study_result["icp_recall"] = icp_metrics["recall"]
            study_result["icp_precision"] = icp_metrics["precision"]
            study_result["icp_time"] = icp_end_time - icp_start_time
            print(f"    ICP Time: {study_result['icp_time']:.4f}s, F1: {icp_metrics['f1_score']:.3f}")

            # 5. Run BHIP
            if D_predictors > 0 : # BHIP requires at least one predictor
                print(f"  Running BHIP (Warmup: {NUM_WARMUP_BHIP}, Samples: {NUM_SAMPLES_BHIP})...")
                X_jax = jnp.array(X_all, dtype=jnp.float32)
                Y_jax = jnp.array(Y_all, dtype=jnp.float32)
                e_jax = jnp.array(env_indices, dtype=jnp.int32)

                bhip_start_time = time.perf_counter()
                kernel_bhip = NUTS(nc_hierarchical_model_general)
                mcmc_bhip = MCMC(kernel_bhip, num_warmup=NUM_WARMUP_BHIP, num_samples=NUM_SAMPLES_BHIP, num_chains=1, progress_bar=False)
                mcmc_bhip.run(
                    random.PRNGKey(int(time.time() * 1000) + n_nodes + rep_idx), # Different key per run, ensure more variability
                    N=N_total,
                    D=D_predictors,
                    E=E_envs,
                    e=e_jax,
                    X=X_jax,
                    y=Y_jax,
                )
                posterior_samples = mcmc_bhip.get_samples()
                bhip_mcmc_end_time = time.perf_counter()
                
                # Perform BHIP Invariance Tests
                mu_pass, beta_pass, pool_pass = invariance_tests(
                    posterior_samples,
                    D=D_predictors,
                    E=E_envs,
                    X_cols=predictor_cols_bhip,
                    printing=False,
                    local_rope="tenth_sd",
                    global_rope="sd",
                    p_thres=0.85,
                    pooling_type="normal"
                )
                bhip_end_time = time.perf_counter()

                # Convert X_col string indices to integers for set operations
                bhip_accepted_set = set(mu_pass).intersection(set(beta_pass)).intersection(set(pool_pass))

                bhip_metrics = calculate_parent_metrics(true_parents, bhip_accepted_set, all_potential_parents)
                study_result["bhip_accepted_set"] = list(bhip_accepted_set)
                study_result["bhip_f1"] = bhip_metrics["f1_score"]
                study_result["bhip_recall"] = bhip_metrics["recall"]
                study_result["bhip_precision"] = bhip_metrics["precision"]
                study_result["bhip_time_mcmc"] = bhip_mcmc_end_time - bhip_start_time
                study_result["bhip_time_total"] = bhip_end_time - bhip_start_time
                print(f"    BHIP Total Time: {study_result['bhip_time_total']:.4f}s (MCMC: {study_result['bhip_time_mcmc']:.4f}s), F1: {bhip_metrics['f1_score']:.3f}")
            else: # No predictors for BHIP (e.g. n_nodes=1, though loop starts at 3, target is n_nodes-1)
                print(f"  Skipping BHIP as D_predictors = 0")
                study_result["bhip_f1"] = np.nan
                study_result["bhip_recall"] = np.nan
                study_result["bhip_precision"] = np.nan
                study_result["bhip_time_mcmc"] = np.nan
                study_result["bhip_time_total"] = np.nan

        except Exception as e:
            print(f"  Error during n_nodes = {n_nodes}: {e}")
            study_result["error"] = str(e)
            # Fill with NaN so DataFrame structure is consistent
            study_result.setdefault("icp_f1", np.nan)
            study_result.setdefault("icp_time", np.nan)
            study_result.setdefault("bhip_f1", np.nan)
            study_result.setdefault("bhip_time_total", np.nan)

        results_list.append(study_result)
        print("-" * 30)
    
    return results_list

def run_single_detailed_experiment():
    """Run a single detailed experiment with increased samples and environments."""
    print("=== Starting Single Detailed Experiment ===")
    
    # Override configuration for this experiment
    N_SAMPLES_PER_ENV = 500
    n_nodes = 15
    NUM_WARMUP_BHIP = 500
    NUM_SAMPLES_BHIP = 1500
    NUM_ENVIRONMENTS = 2
    
    target_node_index = n_nodes - 1
    all_potential_parents = set(range(n_nodes - 1))
    D_predictors = n_nodes - 1

    study_result = {"num_nodes": n_nodes}

    try:
        # 1. Generate DAG
        A = generate_random_dag_and_check(n_nodes, EDGE_PROBABILITY)
        true_parents = set(np.where(A[:, target_node_index] == 1)[0])
        study_result["true_parents"] = list(true_parents)
        print(f"True Parents for target {target_node_index}: {true_parents}")

        # 2. Create SCM
        W = A * np.random.uniform(1, 5, A.shape)
        noise_mean_bounds = (0, 0.1)
        noise_variance_bounds = (0.1, 0.2)
        scm = sempler.LGANM(W, noise_mean_bounds, noise_variance_bounds)

        # 3. Generate Data (Obs + Int)
        data_envs_np_icp = []
        
        # Observational Data (Environment 0)
        obs_data = scm.sample(n=N_SAMPLES_PER_ENV)
        data_envs_np_icp.append(obs_data)

        # Interventional Data (Environments 1-12)
        for env_idx in range(1, NUM_ENVIRONMENTS):
            possible_intervention_nodes = list(range(n_nodes - 1))
            if not possible_intervention_nodes:
                intervention_node = 0
            else:
                intervention_node = np.random.choice(possible_intervention_nodes)
            
            intervention = {intervention_node: (np.random.uniform(3,7), np.random.uniform(0.01, 0.1))}
            study_result[f"intervention_node_env_{env_idx}"] = intervention_node
            int_data = scm.sample(n=N_SAMPLES_PER_ENV, do_interventions=intervention)
            data_envs_np_icp.append(int_data)

        # Prepare data for BHIP
        X_all_list = []
        Y_all_list = []
        env_indices_list = []

        for env_idx, env_data in enumerate(data_envs_np_icp):
            X_env = np.delete(env_data, target_node_index, axis=1)
            Y_env = env_data[:, target_node_index]
            
            X_all_list.append(X_env)
            Y_all_list.append(Y_env)
            env_indices_list.append(np.full(N_SAMPLES_PER_ENV, env_idx))

        X_all = np.concatenate(X_all_list, axis=0)
        Y_all = np.concatenate(Y_all_list, axis=0)
        env_indices = np.concatenate(env_indices_list, axis=0)

        N_total = X_all.shape[0]
        E_envs = NUM_ENVIRONMENTS
        predictor_cols_bhip = [f"X{j}" for j in range(D_predictors)]

        # 4. Run ICP
        print(f"Running ICP (alpha={ALPHA_ICP})...")
        icp_start_time = time.perf_counter()
        result_icp = icp.fit(data=data_envs_np_icp, target=target_node_index, alpha=ALPHA_ICP, verbose=False, color=False)
        icp_end_time = time.perf_counter()
        
        icp_accepted_set = set(result_icp.estimate)
        icp_metrics = calculate_parent_metrics(true_parents, icp_accepted_set, all_potential_parents)
        study_result["icp_accepted_set"] = list(icp_accepted_set)
        study_result["icp_f1"] = icp_metrics["f1_score"]
        study_result["icp_recall"] = icp_metrics["recall"]
        study_result["icp_precision"] = icp_metrics["precision"]
        study_result["icp_time"] = icp_end_time - icp_start_time
        print(f"ICP Time: {study_result['icp_time']:.4f}s, F1: {icp_metrics['f1_score']:.3f}")

        # 5. Run BHIP
        if D_predictors > 0:
            print(f"Running BHIP (Warmup: {NUM_WARMUP_BHIP}, Samples: {NUM_SAMPLES_BHIP})...")
            X_jax = jnp.array(X_all, dtype=jnp.float32)
            Y_jax = jnp.array(Y_all, dtype=jnp.float32)
            e_jax = jnp.array(env_indices, dtype=jnp.int32)

            bhip_start_time = time.perf_counter()
            kernel_bhip = NUTS(nc_hierarchical_model_general)
            mcmc_bhip = MCMC(kernel_bhip, num_warmup=NUM_WARMUP_BHIP, num_samples=NUM_SAMPLES_BHIP, num_chains=1, progress_bar=False)
            mcmc_bhip.run(
                random.PRNGKey(int(time.time() * 1000) + n_nodes),
                N=N_total,
                D=D_predictors,
                E=E_envs,
                e=e_jax,
                X=X_jax,
                y=Y_jax,
            )
            posterior_samples = mcmc_bhip.get_samples()
            bhip_mcmc_end_time = time.perf_counter()
            
            mu_pass, beta_pass, pool_pass = invariance_tests(
                posterior_samples,
                D=D_predictors,
                E=E_envs,
                X_cols=predictor_cols_bhip,
                printing=False,
                local_rope="tenth_sd",
                global_rope="sd",
                p_thres=0.85,
                pooling_type="normal"
            )
            bhip_end_time = time.perf_counter()

            bhip_accepted_set = set(mu_pass).intersection(set(beta_pass)).intersection(set(pool_pass))

            bhip_metrics = calculate_parent_metrics(true_parents, bhip_accepted_set, all_potential_parents)
            study_result["bhip_accepted_set"] = list(bhip_accepted_set)
            study_result["bhip_f1"] = bhip_metrics["f1_score"]
            study_result["bhip_recall"] = bhip_metrics["recall"]
            study_result["bhip_precision"] = bhip_metrics["precision"]
            study_result["bhip_time_mcmc"] = bhip_mcmc_end_time - bhip_start_time
            study_result["bhip_time_total"] = bhip_end_time - bhip_start_time
            print(f"BHIP Total Time: {study_result['bhip_time_total']:.4f}s (MCMC: {study_result['bhip_time_mcmc']:.4f}s), F1: {bhip_metrics['f1_score']:.3f}")
        else:
            print(f"Skipping BHIP as D_predictors = 0")
            study_result["bhip_f1"] = np.nan
            study_result["bhip_recall"] = np.nan
            study_result["bhip_precision"] = np.nan
            study_result["bhip_time_mcmc"] = np.nan
            study_result["bhip_time_total"] = np.nan

    except Exception as e:
        print(f"Error during experiment: {e}")
        study_result["error"] = str(e)
        study_result.setdefault("icp_f1", np.nan)
        study_result.setdefault("icp_time", np.nan)
        study_result.setdefault("bhip_f1", np.nan)
        study_result.setdefault("bhip_time_total", np.nan)

    # Print detailed results
    print("\n=== Detailed Experiment Results ===")
    print(f"Number of Nodes: {n_nodes}")
    print(f"True Parents: {study_result['true_parents']}")
    print(f"ICP Accepted Set: {study_result['icp_accepted_set']}")
    print(f"BHIP Accepted Set: {study_result['bhip_accepted_set']}")
    print(f"ICP F1 Score: {study_result['icp_f1']:.3f}")
    print(f"BHIP F1 Score: {study_result['bhip_f1']:.3f}")
    print(f"ICP Time: {study_result['icp_time']:.4f}s")
    print(f"BHIP Total Time: {study_result['bhip_time_total']:.4f}s")

    return study_result

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run computational study in either parallel or detailed mode')
    parser.add_argument('mode', choices=['parallel', 'detailed'], 
                      help='Mode to run the study in: parallel for multiple replications, detailed for single detailed run')
    args = parser.parse_args()

    if args.mode == 'parallel':
        print(f"Starting computational study for n_nodes in {list(NUM_NODES_RANGE)} with {NUM_REPLICATIONS} replications each...")
        print(f"Samples per environment: {N_SAMPLES_PER_ENV}")
        print("-" * 50)

        # Use fixed number of processes instead of cpu_count
        num_processes = min(MAX_PARALLEL_JOBS, NUM_REPLICATIONS)
        print(f"Using {num_processes} processes for parallel execution")

        # Create a pool of workers and run replications in parallel
        with Pool(processes=num_processes) as pool:
            all_results = pool.map(run_single_replication, range(NUM_REPLICATIONS))

        # Flatten the results list
        results_list = [item for sublist in all_results for item in sublist]

        # --- Final Summary ---
        print("\n--- Overall Metrics Summary (Averaged over Replications) ---")
        results_df = pd.DataFrame(results_list)

        # Ensure all relevant columns exist even if some runs failed partially
        cols_to_check = [
            "replication_id", "num_nodes", "icp_f1", "icp_time", "icp_recall", "icp_precision",
            "bhip_f1", "bhip_time_total", "bhip_time_mcmc", "bhip_recall", "bhip_precision",
            "true_parents", "icp_accepted_set", "bhip_accepted_set", "intervention_node", "error"
        ]
        for col in cols_to_check:
            if col not in results_df.columns:
                results_df[col] = np.nan

        # Select and reorder columns for final display
        display_cols = [
            "replication_id", "num_nodes", "icp_f1", "icp_time", "icp_recall", "icp_precision",
            "bhip_f1", "bhip_time_total", "bhip_recall", "bhip_precision"
        ]
        final_display_cols = [col for col in display_cols if col in results_df.columns]

        # Calculate and print averaged results
        if not results_df.empty:
            # Convert relevant columns to numeric, coercing errors to NaN
            numeric_cols = ["icp_f1", "icp_time", "icp_recall", "icp_precision",
                            "bhip_f1", "bhip_time_total", "bhip_time_mcmc", "bhip_recall", "bhip_precision"]
            for col in numeric_cols:
                if col in results_df.columns:
                    results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

            # Group by num_nodes and calculate mean and std
            averaged_results = results_df.groupby("num_nodes")[numeric_cols].mean()
            std_results = results_df.groupby("num_nodes")[numeric_cols].std()

            # For a cleaner display, rename columns for standard deviation
            std_results.columns = [f"{col}_std" for col in std_results.columns]

            print("\nAveraged Metrics:")
            print(averaged_results.to_string(float_format="%.3f"))

            print("\nStandard Deviations of Metrics:")
            print(std_results.to_string(float_format="%.3f"))

            # Save results to files
            try:
                results_df.to_csv("full_results_data2.csv", index=False)
                print("\nFull results data saved to full_results_data.csv")
                with open("summary_metrics.txt", "w") as f:
                    f.write("Averaged Metrics:\n")
                    f.write(averaged_results.to_string(float_format="%.3f"))
                    f.write("\n\nStandard Deviations of Metrics:\n")
                    f.write(std_results.to_string(float_format="%.3f"))
                print("Summary metrics saved to summary_metrics.txt")
            except Exception as e:
                print(f"Error saving results to file: {e}")

            # Plotting
            print("\nGenerating plots...")
            
            # Time plot
            plt.figure(figsize=(14, 8))
            plot_data_icp = results_df[['num_nodes', 'icp_time']].copy()
            plot_data_icp['method'] = 'ICP'
            plot_data_icp.rename(columns={'icp_time': 'time'}, inplace=True)

            plot_data_bhip = results_df[['num_nodes', 'bhip_time_total']].copy()
            plot_data_bhip['method'] = 'BHIP'
            plot_data_bhip.rename(columns={'bhip_time_total': 'time'}, inplace=True)

            boxplot_df = pd.concat([plot_data_icp, plot_data_bhip], ignore_index=True)
            boxplot_df['time'] = pd.to_numeric(boxplot_df['time'], errors='coerce')
            boxplot_df.dropna(subset=['time'], inplace=True)

            if not boxplot_df.empty:
                sns.boxplot(x='num_nodes', y='time', hue='method', data=boxplot_df, palette="Set2")
                sns.stripplot(x='num_nodes', y='time', hue='method', data=boxplot_df, 
                            size=4, color=".4", dodge=True, alpha=0.7, legend=False)
                
                plt.yscale('log')
                plt.xlabel('Number of Nodes', fontsize=14)
                plt.ylabel('Computational Time (s) - Log Scale', fontsize=14)
                plt.title('Computational Time Distribution: ICP vs BHIP', fontsize=16, pad=20)
                
                handles, labels = plt.gca().get_legend_handles_labels()
                num_methods = boxplot_df['method'].nunique()
                plt.legend(handles[:num_methods], labels[:num_methods], title='Method', fontsize=12, title_fontsize=13)
                
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                plt.savefig("computational_time_boxplot2.png", bbox_inches='tight', dpi=300)
                plt.close()

        else:
            print("No results to display or save.")

    else:  # detailed mode
        print("Starting single detailed experiment...")
        print("-" * 50)

        # Run the single detailed experiment
        result = run_single_detailed_experiment()

        # Save results to file
        try:
            results_df = pd.DataFrame([result])
            results_df.to_csv("detailed_experiment_results.csv", index=False)
            print("\nResults saved to detailed_experiment_results.csv")
        except Exception as e:
            print(f"Error saving results to file: {e}")

        # Create plots for detailed mode
        plt.figure(figsize=(14, 8))
        
        # Create subplots for time and F1 score
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Time plot
        methods = ['ICP', 'BHIP']
        times = [result['icp_time'], result['bhip_time_total']]
        ax1.bar(methods, times, color=['#2ecc71', '#3498db'])
        ax1.set_yscale('log')
        ax1.set_ylabel('Computational Time (s) - Log Scale', fontsize=12)
        ax1.set_title('Computational Time: ICP vs BHIP', fontsize=14, pad=20)
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        # F1 Score plot
        f1_scores = [result['icp_f1'], result['bhip_f1']]
        ax2.bar(methods, f1_scores, color=['#2ecc71', '#3498db'])
        ax2.set_ylim(0, 1)  # F1 score is between 0 and 1
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('F1 Score: ICP vs BHIP', fontsize=14, pad=20)
        ax2.grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("detailed_experiment_plots.png", bbox_inches='tight', dpi=300)
        plt.close()

    print("\nExperiment finished.") 