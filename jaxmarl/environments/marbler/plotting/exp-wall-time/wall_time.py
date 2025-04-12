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
            history = run.history(samples=10000, keys=['_runtime'].extend(metric))

            run_data = pd.DataFrame(history)
            run_data['run_path'] = path
            run_data['run_name'] = run.name
            run_data['run_type'] = 'JaxRobotarium' if 'jax' in path else 'MARBLER'
            run_data['return'] = run_data[metric[0]] * 100 if 'jax' in path else run_data[metric[0]] / 4
            run_data['_runtime'] = run_data['_runtime'] - run_data['_runtime'].min()
            run_data['timestep'] = run_data[metric[1]]

            all_run_data.append(run_data)
        except Exception as e:
            print(f"Failed to fetch run at {path}: {e}")

    return pd.concat(all_run_data, ignore_index=True)

def get_from_wandb(name, run_paths, metrics):
    df = fetch_wandb_data(run_paths, metrics)
    filename = f"{name}.pkl" 
    save_dataframe(df, filename)
    return df

def plot_metric_over_wall_time(df, title, name, metric, swap_axes=False, legend=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Find the minimum max wall_time across all runs
    df = df[df[metric].notnull()]
    min_max_wall_time = df.groupby('run_path')['_runtime'].max().min()

    extrapolated_rows = []

    temp_df = df[df['_runtime'] <= min_max_wall_time]
    for (_, temp_group), (_, group) in zip(temp_df.groupby('run_path'), df.groupby('run_path')):
        group_sorted = group[group.notna()].sort_values('_runtime')
        max_runtime = temp_group['_runtime'].max()

        if max_runtime < min_max_wall_time:
            # Find the last point before and the first point after min_max_wall_time
            before = group_sorted[group_sorted['_runtime'] <= min_max_wall_time]
            before = before[before[metric].notna()].iloc[-1]
            after_candidates = group_sorted[group_sorted['_runtime'] > min_max_wall_time]
            after_candidates = after_candidates[after_candidates[metric].notna()]

            if not after_candidates.empty:
                after = after_candidates.iloc[0]

                t1, y1 = before['_runtime'], before[metric]
                t2, y2 = after['_runtime'], after[metric]

                slope = (y2 - y1) / (t2 - t1)
                extrapolated_y = y1 + slope * (min_max_wall_time - t1)
            else:
                # No point after min_max_wall_time, fallback to constant value
                extrapolated_y = before[metric]

            extrapolated_row = before.copy()
            extrapolated_row['_runtime'] = min_max_wall_time
            extrapolated_row[metric] = extrapolated_y
            extrapolated_rows.append(extrapolated_row)

    # Combine with the original DataFrame
    if extrapolated_rows:
        df = pd.concat([df, pd.DataFrame(extrapolated_rows)], ignore_index=True)

    # Now filter
    df_clipped = df[df['_runtime'] <= min_max_wall_time]

    # Plot
    plt.figure(figsize=(8,6))
    plt.rc('font', size=18)
    if swap_axes:
        sns.lineplot(data=df_clipped, x=metric, y='_runtime', hue='run_type', legend=legend, linewidth=2)
        plt.ylabel("Wall Time (s)")
        plt.xlabel(metric.capitalize())
    else:
        sns.lineplot(data=df_clipped, x='_runtime', y=metric, hue='run_type', legend=legend, linewidth=2)
        plt.xlabel("Wall Time (s)")
        plt.ylabel(metric.capitalize())
    plt.title(title)
    if legend:
        plt.legend(title=None)
    plt.tight_layout()
    plt.savefig(f'{name}.png')

if __name__ == "__main__":
    # Example usage

    # DISCOVERY
    run_paths = [
        "star-lab-gt/jax-marbler/kc777sv5",
        "star-lab-gt/CASH-MARBLER/04bxmm3z"
    ]
    title = 'QMIX / Discovery'
    name = "qmix-discovery"
    metrics = [["returned_episode_returns", "env_step"], ["return_mean", "_step"]]  # specify the metric key(s)
    df = load_dataframe(f"{name}.pkl")
    if df is None:
        df = get_from_wandb(name, run_paths, metrics)

    plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=False)
    plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", swap_axes=True, legend=False)

    # MT
    run_paths = [
        "star-lab-gt/jax-marbler/gjg6c7uc",
        "star-lab-gt/CASH-MARBLER/rpudekeq"
    ]
    metrics = [["returned_episode_returns", "env_step"], ["return_mean", "_step"]] # specify the metric key(s)
    title = 'QMIX / Material Transport'
    name = "qmix-mt"
    df = load_dataframe(f"{name}.pkl")
    if df is None:
        df = get_from_wandb(name, run_paths, metrics)

    plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=False)
    plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", swap_axes=True, legend=False)

    # WAREHOUSE
    run_paths = [
        "star-lab-gt/jax-marbler/p8gk0xo3",
        "star-lab-gt/CASH-MARBLER/cmv6uv67"
    ]
    metrics = [["returned_episode_returns", "env_step"], ["return_mean", "_step"]]  # specify the metric key(s)
    title = 'QMIX / Warehouse'
    name = "qmix-warehouse"
    df = load_dataframe(f"{name}.pkl")
    if df is None:
        df = get_from_wandb(name, run_paths, metrics)

    plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=False)
    plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", swap_axes=True, legend=False)

    # ARCTIC TRANSPORT
    run_paths = [
        "star-lab-gt/jax-marbler/6gxw1hvp",
        "star-lab-gt/CASH-MARBLER/qogkkz1f"
    ]
    metrics = [["returned_episode_returns", "env_step"], ["return_mean", "_step"]]  # specify the metric key(s)
    title = 'QMIX / Arctic Transport'
    name = "qmix-arctic-transport"
    df = load_dataframe(f"{name}.pkl")
    if df is None:
        df = get_from_wandb(name, run_paths, metrics)

    plot_metric_over_wall_time(df, title=title, name=f"{name}-return", metric="return", legend=True)
    plot_metric_over_wall_time(df, title=title, name=f"{name}-timestep", metric="timestep", swap_axes=True, legend=False)