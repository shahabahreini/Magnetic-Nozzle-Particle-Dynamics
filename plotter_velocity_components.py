import matplotlib.pyplot as plt

def plot_velocity_components(time, v_parallel_B, v_perpendicular_B, title, subtitle=None):
    """
    Plots the parallel and perpendicular velocity components against time.

    Parameters:
    - time: Array-like sequence of time values.
    - v_parallel_B: Array-like sequence of parallel velocity component magnitudes.
    - v_perpendicular_B: Array-like sequence of perpendicular velocity component magnitudes.
    - title: Title of the plot.
    - subtitle: Optional subtitle of the plot.
    """
    # Try to use the desired style, fall back if not available
    try:
        plt.style.use('tableau-colorblind10')
    except OSError:
        plt.style.use('ggplot')  # Fallback to an available style

    # Plotting logic
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the figure size to reduce white space
    ax.plot(time, v_parallel_B, label=r'$v_{\parallel}$', marker='o', linestyle='-', linewidth=0.5, markersize=4, color='#1f77b4')
    ax.plot(time, v_perpendicular_B, label=r'$v_{\perp}$', marker='x', linestyle='--', linewidth=0.5, markersize=4, color='#ff7f0e')

    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Velocity Component Value', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=3)  # Adjusted padding for title

    if subtitle:
        ax.text(0.5, 1.05, subtitle, transform=ax.transAxes, fontsize=9, ha='center')  # Adjusted position for subtitle
    
    # Enhance the legend by placing it outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9, frameon=True, shadow=True)

    # Add gridlines (both major and minor)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()

    # Optimize layout to accommodate the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin

    # Show the plot
    plt.show()

# Example usage:
# plot_velocity_components(time_data, v_parallel_data, v_perpendicular_data, "Main Title", "Subtitle")
