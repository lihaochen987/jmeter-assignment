import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def create_box_dot_plot(csv_file, output_file=None, width=14, height=10):
    """
    Creates a combined box and dot plot for JMeter sampler latencies with summary statistics.

    Parameters:
    -----------
    csv_file : str
        Path to the JMeter CSV file
    output_file : str, optional
        Path to save the output image. If None, the plot will be displayed.
    width : int, optional
        Width of the plot in inches. Default is 14.
    height : int, optional
        Height of the plot in inches. Default is 10.

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
        The generated plot
    df_stats : pandas DataFrame
        Summary statistics for each sampler
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

    # Calculate statistics for each sampler
    stats = df.groupby('label')['Latency'].agg([
        ('count', 'count'),
        ('min', 'min'),
        ('q1', lambda x: x.quantile(0.25)),
        ('median', 'median'),
        ('mean', 'mean'),
        ('q3', lambda x: x.quantile(0.75)),
        ('max', 'max'),
        ('std', 'std'),
        ('90p', lambda x: x.quantile(0.90)),
        ('95p', lambda x: x.quantile(0.95)),
        ('99p', lambda x: x.quantile(0.99))
    ]).reset_index()

    # Sort samplers by median latency for better visualization
    stats = stats.sort_values('median')
    sampler_order = stats['label'].tolist()

    df['label_sorted'] = pd.Categorical(
        df['label'],
        categories=sampler_order,
        ordered=True
    )

    # Create the figure with a grid layout
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)

    # Plot area for box-dot plot
    ax_plot = fig.add_subplot(gs[0])

    # Create a combined box and strip plot
    sns.boxplot(x='label_sorted', y='Latency', data=df, ax=ax_plot,
                color='lightgray', width=0.5)

    sns.stripplot(x='label_sorted', y='Latency', data=df, ax=ax_plot,
                  jitter=True, alpha=0.5, size=4)

    # Customize the plot
    ax_plot.set_title('JMeter Sampler Latency: Box Plot with Individual Data Points', fontsize=14)
    ax_plot.set_xlabel('')  # Remove x label as it will be in the table
    ax_plot.set_ylabel('Latency (ms)', fontsize=12)
    ax_plot.set_xticklabels([])  # Hide x tick labels as they will be in the table
    ax_plot.grid(axis='y', linestyle='--', alpha=0.7)

    # Create a table for the statistics
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')  # Hide the axes

    # Format statistics for display
    display_stats = stats.copy()

    # Round numeric columns to 1 decimal place
    numeric_cols = display_stats.columns.drop('label')
    display_stats[numeric_cols] = display_stats[numeric_cols].round(1)

    # Convert to string for display formatting
    display_stats = display_stats.astype(str)

    # Select columns for the table (to avoid it being too wide)
    table_cols = ['label', 'count', 'min', 'q1', 'median', 'mean', 'q3', 'max', '95p']
    table_data = display_stats[table_cols].values.tolist()

    # Add the column headers
    column_labels = ['Sampler', 'Count', 'Min', 'Q1', 'Median', 'Mean', 'Q3', 'Max', '95th %']

    # Create the table
    table = ax_table.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.2] + [0.1] * (len(column_labels) - 1)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # Make rows a bit taller

    # Color alternating rows for readability
    for i, row in enumerate(table_data):
        for j in range(len(column_labels)):
            cell = table[(i + 1, j)]  # +1 to skip header row
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')

    # Header row styling
    for j, col in enumerate(column_labels):
        cell = table[(0, j)]
        cell.set_facecolor('#c0c0c0')
        cell.set_text_props(weight='bold')

    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig, ax_plot, stats

def create_latency_elapsed_scatter(csv_file, output_file=None, width=12, height=10,
                                   color_by_sampler=True, add_identity_line=True):
    """
    Creates a scatter plot comparing Latency vs Elapsed time from JMeter results.

    Parameters:
    -----------
    csv_file : str
        Path to the JMeter CSV file
    output_file : str, optional
        Path to save the output image. If None, the plot will be displayed.
    width : int, optional
        Width of the plot in inches. Default is 12.
    height : int, optional
        Height of the plot in inches. Default is 10.
    color_by_sampler : bool, optional
        Whether to color points by sampler type. Default is True.
    add_identity_line : bool, optional
        Whether to add a y=x reference line. Default is True.

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

    # Convert columns to numeric
    df['Latency'] = pd.to_numeric(df['Latency'], errors='coerce')
    df['elapsed'] = pd.to_numeric(df['elapsed'], errors='coerce')

    # Drop rows with missing values
    df = df.dropna(subset=['Latency', 'elapsed'])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width, height))

    if color_by_sampler:
        # Create a scatter plot with points colored by sampler
        samplers = df['label'].unique()

        # Create a color palette with enough distinct colors
        if len(samplers) <= 10:
            palette = sns.color_palette("tab10", len(samplers))
        else:
            palette = sns.color_palette("husl", len(samplers))

        # Plot each sampler with a different color
        for i, sampler in enumerate(samplers):
            sampler_data = df[df['label'] == sampler]
            plt.scatter(sampler_data['Latency'], sampler_data['elapsed'],
                        label=sampler, color=palette[i], alpha=0.7)

        plt.legend(title='Sampler', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Simple scatter plot without coloring by sampler
        plt.scatter(df['Latency'], df['elapsed'], alpha=0.7)

    # Add identity line (y=x) if requested
    if add_identity_line:
        max_val = max(df['Latency'].max(), df['elapsed'].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')

    # Set equal axis limits for better comparison
    max_limit = max(df['Latency'].max(), df['elapsed'].max()) * 1.05
    plt.xlim(0, max_limit)
    plt.ylim(0, max_limit)

    # Calculate correlation coefficient
    correlation = df['Latency'].corr(df['elapsed'])

    # Calculate average time for transfer (elapsed - latency)
    df['transfer_time'] = df['elapsed'] - df['Latency']
    avg_transfer = df['transfer_time'].mean()

    # Add annotations with statistics
    stats_text = f"Correlation: {correlation:.3f}\nAvg Transfer Time: {avg_transfer:.2f} ms"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

    # Customize the plot
    plt.title('JMeter Performance: Latency vs Elapsed Time', fontsize=14)
    plt.xlabel('Latency (ms) - Time to First Byte', fontsize=12)
    plt.ylabel('Elapsed Time (ms) - Total Request Duration', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensure the legend doesn't overlap with the plot
    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    return fig, ax


# Example usage:
if __name__ == "__main__":
    # File path for your JMeter data
    jmeter_file = 'output.csv'  # Change to your file path

    # Create the box-dot plot and save it
    box_fig, box_ax, box_stats_df = create_box_dot_plot(jmeter_file, 'jmeter_latency_box_dot.png')
    plt.close(box_fig)  # Close the figure to avoid displaying it immediately

    # Create the scatter plot and save it
    scatter_fig, scatter_ax = create_latency_elapsed_scatter(jmeter_file, 'jmeter_latency_vs_elapsed.png')
    plt.close(scatter_fig)  # Close the figure