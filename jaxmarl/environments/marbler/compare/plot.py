import matplotlib.pyplot as plt
import seaborn as sns

def plot_bars(names, values, colors, threshold, threshold_label):
    fig, ax = plt.subplots()
    
    # Plot bars
    bars = ax.bar(names, values, color=colors)
    
    # Add threshold line
    ax.axhline(y=threshold, color='black', linestyle='dashed', linewidth=1)
    
    # Add threshold label
    ax.text(
        x=len(names) - 1, 
        y=threshold + (max(values) * 0.05), 
        s=threshold_label, 
        fontsize=8, 
        color='black', 
        ha='right'
    )

    ax.set_ylabel("Elapsed Time (s)")
    
    # Show plot
    plt.savefig('scale.png')

# Example usage
names = ["1 env, 1 seed", "8 envs, 1 seed", "8 envs, 10 seeds"]
values = [1539.960, 283.541, 719.635]
colors = sns.color_palette(n_colors=3)
thresh = 11156.984
thresh_label = "MARBLER (1 env, 1 seed)"

plot_bars(names, values, colors, thresh, thresh_label)