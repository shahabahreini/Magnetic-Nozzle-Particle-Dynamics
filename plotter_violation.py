import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

config = {
  'toImageButtonOptions': {
    'format': 'png',  # one of png, svg, jpeg, webp
    'filename': 'custom_image',
    'height': 500,
    'width': 700,
    'scale': 4  # Multiply title/legend/axis/canvas sizes by this factor
  }
}

def moving_average(data, window_size):
    """Calculate the moving average for a given data and window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def find_crossing_point(x_values, y_values, threshold):
    """Find the exact crossing point where y_values cross the threshold."""
    for i in range(1, len(y_values)):
        if (y_values[i-1] < threshold and y_values[i] > threshold) or (y_values[i-1] > threshold and y_values[i] < threshold):
            # Linear interpolation to find the crossing point
            slope = (y_values[i] - y_values[i-1]) / (x_values[i] - x_values[i-1])
            intercept = y_values[i-1] - slope * x_values[i-1]
            crossing_x = (threshold - intercept) / slope
            return crossing_x
    return None

def load_and_calculate_variation(mu, x, eps, threshold=0.01, window_size=5):
    # Convert mu_values to a numpy array for calculation
    mu_values = np.array(mu)

    # If x_values are not provided, use the index as x_values
    if x is None:
        x_values = np.arange(len(mu_values))
    else:
        x_values = np.array(x)

    # Calculate delta(mu) and delta(mu)/mu
    delta_mu = abs(np.diff(mu_values))
    delta_mu_over_mu = delta_mu / mu_values[:-1]

    # Calculate the moving average for delta(mu)/mu
    delta_mu_over_mu_ma = moving_average(delta_mu_over_mu, window_size)
    
    # Adjust x_values for moving average (as it shortens the array length)
    x_values_ma = x_values[:len(delta_mu_over_mu_ma)]

    # Find the first violation index where delta(mu)/mu exceeds the threshold
    violation_indices = np.where(np.abs(delta_mu_over_mu) > threshold)[0]
    violation_index = violation_indices[0] if len(violation_indices) > 0 else None

    # Find the exact crossing point for the moving average
    ma_crossing_x = find_crossing_point(x_values_ma, delta_mu_over_mu_ma, threshold)

    # Create the main line plot for delta(mu)/mu
    trace = go.Scatter(
        x=x_values[:-1],
        y=delta_mu_over_mu,
        mode='lines+markers',
        marker=dict(size=4, color='blue'),
        line=dict(width=1),
        name=r'$\Delta(\mu)/\mu$'
    )

    # Create the moving average plot
    ma_trace = go.Scatter(
        x=x_values_ma,
        y=delta_mu_over_mu_ma,
        mode='lines',
        line=dict(color='green', dash='dash', width=3),
        name='Moving Average'
    )

    # Create a horizontal line for the threshold
    threshold_line = go.Scatter(
        x=x_values,
        y=[threshold] * len(x_values),
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name=f'Limit: {threshold}'
    )

    # Create a vertical line for the first violation in delta(mu)/mu
    violation_line = None
    if violation_index is not None:
        violation_line = go.Scatter(
            x=[x_values[violation_index], x_values[violation_index]],
            y=[min(delta_mu_over_mu), max(delta_mu_over_mu)],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Violation at x = {x_values[violation_index]}'
        )

    # Create an orange line at the crossing point of the moving average
    ma_crossing_line = None
    if ma_crossing_x is not None:
        ma_crossing_line = go.Scatter(
            x=[ma_crossing_x, ma_crossing_x],
            y=[min(delta_mu_over_mu_ma), max(delta_mu_over_mu_ma)],
            mode='lines',
            line=dict(color='orange', dash='dash'),
            name=f'MA Crossing at x = {ma_crossing_x:.2f}'
        )

    # Combine traces
    data = [trace, ma_trace, threshold_line]
    if violation_line is not None:
        data.append(violation_line)
    if ma_crossing_line is not None:
        data.append(ma_crossing_line)

    # Layout for the plot
    layout = go.Layout(
        title=dict(
            text=r'Relative Change Ratio and Moving Average (MA)',
            font=dict(size=18)  # Increase font size for title
        ),
        annotations=[
            dict(
                text=rf"$\epsilon={eps}$",
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=14)
            )
        ],
        xaxis=dict(
            title=r'$\tau$',
            titlefont=dict(size=25)  # Increase font size for x-axis label
        ),
        yaxis=dict(
            title=r'$\Delta(\mu)/\mu$',
            titlefont=dict(size=25)  # Increase font size for y-axis label
        ),
        legend=dict(x=0, y=1, bgcolor='rgba(255, 255, 255, 0)'),
        hovermode='closest'
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Show the figure with the configuration
    fig.show(config=config)

# Example usage:
if __name__ == "__main__":
    mu_values = [5e-7, 5e-7, 5e-7, 4.9e-7, 4.7e-7, 4.8e-7, 4.6e-7]  # Example mu values
    x_values = [1, 2, 3, 4, 5, 6, 7]  # Example x values (optional)

    load_and_calculate_variation(mu_values, x_values)
