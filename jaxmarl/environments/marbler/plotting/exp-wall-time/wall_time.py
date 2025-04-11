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

def fetch_wandb_data(run_paths, metrics):
    api = wandb.Api(timeout=60)
    all_run_data = []

    for path, metric in zip(run_paths, metrics):
        try:
            run = api.run(path)
            keys = [metric, '_runtime']
            print(f"Fetching: {run.name} with metrics {keys} from {path}")
            history = run.history(samples=7000, keys=[metric, '_runtime'])

            run_data = pd.DataFrame(history)
            run_data['run_path'] = path
            run_data['run_name'] = run.name
            run_data['run_type'] = 'Jax' if 'jax' in path else 'Python'
            run_data['return'] = run_data[metric] * 100 if 'jax' in path else run_data[metric] / 4
            run_data['_runtime'] = run_data['_runtime'] - run_data['_runtime'].min()

            all_run_data.append(run_data)
        except Exception as e:
            print(f"Failed to fetch run at {path}: {e}")

    return pd.concat(all_run_data, ignore_index=True)

def get_from_wandb(name, run_paths, metrics):
    df = fetch_wandb_data(run_paths, metrics)
    filename = f"{name}.pkl" 
    save_dataframe(df, filename)
    return df

def plot_metric_over_wall_time(df, title, name, metric):
    # Find the minimum max wall_time across all runs
    min_max_wall_time = df.groupby('run_path')['_runtime'].max().min()

    # Clip all data to this wall time
    df_clipped = df[df['_runtime'] <= min_max_wall_time]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_clipped, x='_runtime', y=metric, hue='run_type')
    plt.title(title)
    plt.xlabel("Wall Time (s)")
    plt.ylabel("Return")
    plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(f'{name}.svg')

if __name__ == "__main__":
    # Example usage
    run_paths = [
        "star-lab-gt/jax-marbler/kc777sv5",
        "star-lab-gt/CASH-MARBLER/04bxmm3z"
    ]
    metrics = ["returned_episode_returns", "return_mean"]  # specify the metric key(s)

    title = 'QMIX / Discovery'
    name = "qmix-discovery"
    df = load_dataframe(f"{name}.pkl")
    if df is None:
        df = get_from_wandb(name, run_paths, metrics)

    plot_metric_over_wall_time(df, title=title, name=name, metric="return")