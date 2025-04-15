import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def save_dataframe(df, filename):
    df.to_pickle(filename)
    print(f"DataFrame saved to {filename}")

def load_dataframe(filename):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print(f"DataFrame loaded from {filename}")
        return df
    else:
        print(f"File {filename} not found.")
        return None

def fetch_wandb_data(project_name, tags, metrics):
    api = wandb.Api(timeout=60)
    runs = api.runs(project_name, {"tags": {"$in": tags}})
    print(f"found {len(runs)} runs with tags {tags}")

    all_run_data = []
    for run in runs:
        # get metrics of run over time
        history = run.scan_history(keys=metrics)

        run_data = pd.DataFrame(history)
        run_data['run_id'] = run.id
        run_data['run_name'] = run.name
        run_data['timestep'] = run_data['env_step']
        run_data['tags'] = ', '.join(run.tags)
        run_data['env_name'] = run.config.get("ENV_NAME")

        print(run_data.head())

        all_run_data.append(run_data)

    return pd.concat(all_run_data, ignore_index=True)

def get_from_wandb(tags, metrics):
    # get runs from wandb
    # NOTE: I've been hardcoding the right things in...

    project_name = "jax-marbler"

    df = fetch_wandb_data(project_name, tags, metrics)
    filename = f"{tags[0]}.pkl" 
    save_dataframe(df, filename)

    print("saved")
    print(df.head())

def smooth_and_downsample(df, y_column, mean_window=50, std_window=50, downsample_factor=10):
    """
    Creates a new dataframe with smoothed and downsampled data, with separate
    smoothing controls for mean and standard deviation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    y_column : str
        Column to analyze
    mean_window : int
        Window size for smoothing the mean
    std_window : int
        Window size for smoothing the standard deviation
    downsample_factor : int
        Factor by which to downsample the data
    """
    smoothed_data = []
    df_copy = df.copy()

    for baseline in df_copy['run_id'].unique():
        baseline_data = df_copy[df_copy['run_id'] == baseline].copy()
        baseline_data = baseline_data.sort_values('timestep')

        # grouped = baseline_data.groupby('timestep')[y_column] # .agg(['mean', 'std']).reset_index()
        max_timesteps = baseline_data['timestep'].max()
        max_timesteps_rows = baseline_data[baseline_data['timestep'] == max_timesteps]
        cols = [y_column, 'run_id', 'run_name']
        print("ALL SEEDS metrics:")
        print(max_timesteps_rows[cols])
        print()

        # Group by timestep to calculate mean and std
        grouped = baseline_data.groupby('timestep')[y_column].agg(['mean', 'std']).reset_index()

        # Smooth mean and std separately
        grouped['smooth_mean'] = grouped['mean'].rolling(
            window=mean_window, min_periods=1, center=True).mean()
        grouped['smooth_std'] = grouped['std'].rolling(
            window=std_window, min_periods=1, center=True).mean()

        # Downsample
        grouped = grouped.iloc[::downsample_factor]

        # Create dataframe with smoothed mean and smoothed std
        smoothed_df = pd.DataFrame({
            'timestep': grouped['timestep'],
            f'{y_column}': grouped['smooth_mean'],
            f'{y_column}_std': grouped['smooth_std'],
            'run_id': baseline
        })

        smoothed_data.append(smoothed_df)

    return pd.concat(smoothed_data)

def plot_comparison_by_env(df_qmix, df_pqn, y_column='test_returned_episode_returns',
                            mean_window=50, std_window=50, downsample_factor=10):
    df_qmix['algorithm'] = 'QMIX'
    df_pqn['algorithm'] = 'PQN'

    combined_df = pd.concat([df_qmix, df_pqn], ignore_index=True)

    env_names = combined_df['env_name'].unique()
    for env in env_names:
        env_df = combined_df[combined_df['env_name'] == env]

        smoothed_data = []

        for (run_id, algo), group in env_df.groupby(['run_id', 'algorithm']):
            group = group.sort_values('timestep')

            grouped = group.groupby('timestep')[y_column].agg(['mean', 'std']).reset_index()

            grouped['smooth_mean'] = grouped['mean'].rolling(
                window=mean_window, min_periods=1, center=True).mean()
            grouped['smooth_std'] = grouped['std'].rolling(
                window=std_window, min_periods=1, center=True).mean()

            grouped = grouped.iloc[::downsample_factor]

            grouped['algorithm'] = algo
            grouped['env_name'] = env

            smoothed_data.append(grouped)

        final_df = pd.concat(smoothed_data, ignore_index=True)

        plt.figure(figsize=(8, 6))
        plt.rc('font', size=18)

        for algo in final_df['algorithm'].unique():
            algo_df = final_df[final_df['algorithm'] == algo]
            plt.plot(algo_df['timestep'], algo_df['smooth_mean'], label=algo)
            plt.fill_between(
                algo_df['timestep'],
                algo_df['smooth_mean'] - algo_df['smooth_std'],
                algo_df['smooth_mean'] + algo_df['smooth_std'],
                alpha=0.3
            )

        plt.title(f"{' '.join(env.split('_')).title()}")
        plt.xlabel("Timestep")
        plt.ylabel("Return")
        plt.legend(title=None)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{env}.png')
        # plt.show()  # Uncomment if you want to view interactively


if __name__ == "__main__":
    tags = ['final-qmix']
    metrics = ['test_returned_episode_returns', 'env_step']
    # get_from_wandb(tags, metrics)
    df_qmix = load_dataframe('final-qmix.pkl')
    # smoothed_df_qmix = smooth_and_downsample(df_qmix, 'test_returned_episode_returns')

    tags = ['final-pqn']
    metrics = ['test_returned_episode_returns', 'env_step']
    # get_from_wandb(tags, metrics)
    df_pqn = load_dataframe('final-pqn.pkl')
    # smoothed_df_pqn = smooth_and_downsample(df_pqn, 'test_returned_episode_returns')

    # generate plots test_returned_episode_returns of qmix vs pqn per scenario with x axis as timestep and y axis as returns
    plot_comparison_by_env(df_qmix, df_pqn)