import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_box_dot_plot(csv_file, output_file=None, width=12, height=8):
    """
    Creates a combined box and dot plot for JMeter sampler latencies.

    Parameters:
    -----------
    csv_file : str
        Path to the JMeter CSV file
    output_file : str, optional
        Path to save the output image. If None, the plot will be displayed.
    width : int, optional
        Width of the plot in inches. Default is 12.
    height : int, optional
        Height of the plot in inches. Default is 8.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The generated plot
    """
    # Read the JMeter CSV file
    df = pd.read_csv(csv_file, sep=',')

    # Filter out Debug Sampler and transaction rows (aggregated stats)
    df = df[~df['label'].str.contains('/')]  # Filter out transaction rows
    df = df[df['label'] != 'Debug Sampler']  # Filter out Debug Sampler

    # Convert latency column to numeric
    df['Latency'] = pd.to_numeric(df['Latency'], errors='coerce')

    # Drop rows with missing latency values
    df = df.dropna(subset=['Latency'])

    # Sort samplers by median latency for better visualization
    sampler_medians = df.groupby('label')['Latency'].median().sort_values()
    df['label_sorted'] = pd.Categorical(
        df['label'],
        categories=sampler_medians.index,
        ordered=True
    )

    # Create the figure
    fig, ax = plt.subplots(figsize=(width, height))

    # Create a combined box and swarm plot
    # First create the box plot for the statistical summary
    sns.boxplot(x='label_sorted', y='Latency', data=df, ax=ax,
                color='lightgray', width=0.5)

    # Overlay with the dot plot showing individual points
    sns.stripplot(x='label_sorted', y='Latency', data=df, ax=ax,
                  jitter=True, alpha=0.5, size=4)

    # Customize the plot
    plt.title('JMeter Sampler Latency: Box Plot with Individual Data Points', fontsize=14)
    plt.xlabel('Sampler', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig, ax


# Example usage:
if __name__ == "__main__":
    # Create the box-dot plot
    fig, ax = create_box_dot_plot('output.csv', 'jmeter_latency_box_dot.png')

    # Display the plot (will be shown only if running in an interactive environment)
    plt.show()
