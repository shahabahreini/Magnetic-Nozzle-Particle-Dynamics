import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mayavi import mlab
import pyvista as pv
import holoviews as hv
from holoviews import opts

# Initialize HoloViews with the Bokeh backend
hv.extension('bokeh')
pv.global_theme.allow_empty_mesh = True

# Enable or disable specific visualizations
MATPLOTLIB_2D_STREAM_CONTOUR = False
MATPLOTLIB_2D_QUIVER = False
MATPLOTLIB_2D_ELECTRIC_CONTOUR = False
MATPLOTLIB_2D_POTENTIAL_CONTOUR = False
MATPLOTLIB_3D_SURFACE = True
PLOTLY_3D_SURFACE = False
MAYAVI_VECTOR_FIELD = False
PYVISTA_STREAMLINES = False
HOLOVIEWS_QUAD_MESH = False

# Physical and plot constants
PHYSICS_PARAMS = {
    "B0": 1.0,  # Magnetic field strength (T)
    "r0": 1.0,  # Reference radius (m)
    "Phi0": 1.0,  # Reference electrostatic potential (V)
    "kappa": 1.0,  # Temperature gradient parameter
    "r_star": 2.0,  # Characteristic radius for temperature profile (m)
}

delta_star = PHYSICS_PARAMS["r0"] / PHYSICS_PARAMS["r_star"]

# Define electric potential and field functions
def Phi(R, Z):
    term1 = PHYSICS_PARAMS["kappa"] * PHYSICS_PARAMS["Phi0"] * (
        1 - delta_star**2 * (1 - Z**2 / (R**2 + Z**2))
    ) * np.log(1 / (R**2 + Z**2))
    term2 = 0.5 * PHYSICS_PARAMS["Phi0"] * (1 - Z**2 / (R**2 + Z**2))
    return term1 + term2

def E_R(R, Z):
    Phi0 = PHYSICS_PARAMS["Phi0"]
    kappa = PHYSICS_PARAMS["kappa"]
    delta_star2 = delta_star**2
    numerator = R * Phi0 * (
        -2 * kappa * R**2 
        + 2 * delta_star2 * kappa * (R**2 - Z**2 * np.log(1 / (R**2 + Z**2)))
        + (1 - 2 * kappa) * Z**2
    )
    denominator = (R**2 + Z**2)**2
    return -numerator / denominator

def E_z(R, Z):
    Phi0 = PHYSICS_PARAMS["Phi0"]
    kappa = PHYSICS_PARAMS["kappa"]
    delta_star2 = delta_star**2
    numerator = -Z * Phi0 * (
        (2 * kappa + 1) * R**2 
        - 2 * delta_star2 * kappa * R**2 * (np.log(1 / (R**2 + Z**2)) + 1)
        + 2 * kappa * Z**2
    )
    denominator = (R**2 + Z**2)**2
    return -numerator / denominator

# Matplotlib 2D Stream and Contour Plot
def matplotlib_2d_stream_contour():
    fig, ax = plt.subplots(figsize=(10, 8))
    R = np.linspace(0.1, 4, 100)
    Z = np.linspace(0.1, 8, 100)
    Z, R = np.meshgrid(Z, R)  # Swap the order of Z and R

    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    E_magnitude = np.sqrt(E_R_values**2 + E_z_values**2)

    # Use Z for x-axis and R for y-axis in streamplot and contour
    streamplot = ax.streamplot(Z, R, E_z_values, E_R_values, color=E_magnitude, cmap="viridis", linewidth=1.5)
    potential = Phi(R, Z)
    contour = ax.contour(Z, R, potential, levels=20, colors="red", linewidths=0.5)
    
    equipotential_line = plt.Line2D([0], [0], color="red", lw=0.5, label="Equipotential Lines")
    electric_field_line = plt.Line2D([0], [0], color="purple", lw=1.5, label="Electric Field Lines")
    ax.legend(handles=[equipotential_line, electric_field_line], loc="upper right")

    plt.colorbar(streamplot.lines, label="Electric field magnitude (V/m)", pad=0.1)
    ax.set_title("Electric Field (2D Streamplot) with Equipotential Lines")
    ax.set_xlabel("Z (m)")  # Change label to Z
    ax.set_ylabel("R (m)")  # Change label to R
    plt.tight_layout()
    plt.show()


# Matplotlib 2D Quiver Plot
def matplotlib_2d_quiver():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Dense grid for potential color map (rotated)
    Z_dense = np.linspace(0.2, 8, 100)
    R_dense = np.linspace(0.2, 4, 100)
    Z_dense, R_dense = np.meshgrid(Z_dense, R_dense)
    potential = Phi(R_dense, Z_dense)
    contourf = ax.contourf(Z_dense, R_dense, potential, levels=50, cmap="coolwarm", alpha=0.6)
    plt.colorbar(contourf, ax=ax, label="Electric Potential (V)")

    # Sparse grid for quiver arrows (rotated)
    Z_sparse = np.linspace(0.5, 8, 30)
    R_sparse = np.linspace(0.5, 4, 30)
    Z_sparse, R_sparse = np.meshgrid(Z_sparse, R_sparse)
    E_R_values = E_R(R_sparse, Z_sparse)
    E_z_values = E_z(R_sparse, Z_sparse)

    # Quiver plot for electric field
    quiver = ax.quiver(Z_sparse, R_sparse, E_z_values, E_R_values, color="black", angles="xy", scale_units="xy", scale=3, alpha=0.5)

    # Create a custom legend entry for Electric Potential Contours using a dummy Line2D object
    from matplotlib.lines import Line2D
    contour_legend = Line2D([0], [0], color="black", linestyle="-", linewidth=0.8, label="Electric Equipotential Lines")

    # Set plot title and labels
    ax.set_title("Electric Field (2D Quiver Plot) with Electric Potential Color Map")
    ax.set_xlabel("Z (m)")
    ax.set_ylabel("R (m)")

    # Add both entries to the legend
    ax.legend(handles=[contour_legend], loc="upper left")

    plt.tight_layout()
    plt.show()
        
def matplotlib_2d_potential_contour():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define grid for contour plot (rotated)
    Z = np.linspace(0.1, 8, 200)
    R = np.linspace(0.1, 4, 200)
    Z, R = np.meshgrid(Z, R)
    potential = Phi(R, Z)

    # Plot filled contours
    contourf = ax.contourf(Z, R, potential, levels=50, cmap="viridis", alpha=0.8)
    colorbar = plt.colorbar(contourf, ax=ax, label="Electric Potential (V)")

    # Overlay contour lines for better visual clarity
    contours = ax.contour(Z, R, potential, levels=10, colors="black", linewidths=0.5)

    # Add labels for contour lines
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f V")

    # Set plot title and axis labels
    ax.set_title("2D Electric Potential Contour Plot", fontsize=14, weight="bold")
    ax.set_xlabel("Z (m)", fontsize=12)
    ax.set_ylabel("R (m)", fontsize=12)

    # Set grid and limits for professional look
    ax.grid(visible=True, linestyle="--", color="grey", alpha=0.3)
    ax.set_xlim(0.1, 8)
    ax.set_ylim(0.1, 4)

    # Adjust layout for a clean look
    plt.tight_layout()
    plt.show()


def matplotlib_2d_electric_field_contour():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define grid for contour plot (rotated)
    Z = np.linspace(0.1, 8, 200)
    R = np.linspace(0.1, 4, 200)
    Z, R = np.meshgrid(Z, R)

    # Calculate electric field components and magnitude
    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    E_magnitude = np.sqrt(E_R_values**2 + E_z_values**2)

    # Plot filled contours for electric field magnitude
    contourf = ax.contourf(Z, R, E_magnitude, levels=50, cmap="plasma", alpha=0.8)
    colorbar = plt.colorbar(contourf, ax=ax, label="Electric Field Magnitude (V/m)")

    # Overlay contour lines for electric field magnitude
    contours = ax.contour(Z, R, E_magnitude, levels=10, colors="black", linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f V/m")

    # Set plot title and axis labels
    ax.set_title("2D Electric Field Magnitude Contour Plot", fontsize=14, weight="bold")
    ax.set_xlabel("Z (m)", fontsize=12)
    ax.set_ylabel("R (m)", fontsize=12)

    # Set grid and limits for professional look
    ax.grid(visible=True, linestyle="--", color="grey", alpha=0.3)
    ax.set_xlim(0.1, 8)
    ax.set_ylim(0.1, 4)

    # Adjust layout for a clean look
    plt.tight_layout()
    plt.show()

    
    
# Matplotlib 3D Surface Plot
def matplotlib_3d_surface():
    if MATPLOTLIB_3D_SURFACE:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        R = np.linspace(0.1, 5, 100)
        Z = np.linspace(0.1, 5, 100)
        R, Z = np.meshgrid(R, Z)
        E_magnitude = np.sqrt(E_R(R, Z)**2 + E_z(R, Z)**2)
        potential = Phi(R, Z)

        surf = ax.plot_surface(R, Z, E_magnitude, cmap="viridis", edgecolor="none", alpha=0.7)
        fig.colorbar(surf, ax=ax, label="Electric field magnitude (V/m)", pad=0.1)
        ax.contour(R, Z, potential, levels=15, offset=E_magnitude.min(), cmap="cool", linestyles="dashed")

        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_zlabel("Electric Field Magnitude (V/m)")
        ax.set_title("Electric Field Magnitude (3D Surface Plot) with Equipotential Contours")
        plt.tight_layout()
        plt.show()

# Plotly Interactive 3D Surface Plot
def plotly_3d_surface():
    if PLOTLY_3D_SURFACE:
        R = np.linspace(0.1, 5, 100)
        Z = np.linspace(0.1, 5, 100)
        R, Z = np.meshgrid(R, Z)
        E_magnitude = np.sqrt(E_R(R, Z)**2 + E_z(R, Z)**2)

        fig = go.Figure(data=[go.Surface(z=E_magnitude, x=R, y=Z, colorscale='Viridis')])
        fig.update_layout(
            title="Electric Field Magnitude",
            scene=dict(
                xaxis_title="R (m)",
                yaxis_title="Z (m)",
                zaxis_title="Field Magnitude (V/m)"
            ),
        )
        fig.show()

# Mayavi 3D Vector Field
def mayavi_vector_field():
    if MAYAVI_VECTOR_FIELD:
        R = np.linspace(-5, 5, 10)
        Z = np.linspace(-5, 5, 10)
        R, Z = np.meshgrid(R, Z)
        E_R_values = E_R(R, Z)
        E_z_values = E_z(R, Z)

        mlab.figure(size=(800, 600))
        mlab.quiver3d(R, Z, np.zeros_like(R), E_R_values, E_z_values, np.zeros_like(R), color=(0, 0, 1))
        mlab.title("Electric Field Vector Field")
        mlab.xlabel("R")
        mlab.ylabel("Z")
        mlab.show()

# PyVista Streamlines
def pyvista_streamlines():
    if PYVISTA_STREAMLINES:
        R = np.linspace(-5, 5, 50)
        Z = np.linspace(-5, 5, 50)
        R, Z = np.meshgrid(R, Z)
        E_R_values = E_R(R, Z)
        E_z_values = E_z(R, Z)
        potential = Phi(R, Z)

        grid = pv.StructuredGrid(R, Z, np.zeros_like(R))
        grid['E_field'] = np.c_[E_R_values.ravel(), E_z_values.ravel(), np.zeros_like(E_R_values).ravel()]
        grid['potential'] = potential.ravel()

        # Create a source point for the streamlines
        source = pv.PolyData([0, 0, 0])

        plotter = pv.Plotter()
        plotter.add_mesh(grid.contour(10, scalars='potential'), cmap='coolwarm', line_width=2)
        streamlines = grid.streamlines_from_source(
            source, vectors='E_field', max_time=100, integration_direction='both'
        )
        plotter.add_mesh(streamlines.tube(radius=0.01), color='blue')
        plotter.add_mesh(grid.outline(), color="k")
        plotter.show()

# Holoviews + Datashader High-Resolution Quiver Plot
def holoviews_vectorfield_plot():
    # Increase the resolution for smoother contours and vector fields
    R = np.linspace(-5, 5, 100)
    Z = np.linspace(-5, 5, 100)
    R, Z = np.meshgrid(R, Z)
    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    potential_values = Phi(R, Z)

    # Prepare vector data for the plot
    vector_data = (R.ravel(), Z.ravel(), E_R_values.ravel(), E_z_values.ravel())
    
    # Create a VectorField plot with adjusted scaling and no colorbar
    vector_field = hv.VectorField(vector_data).opts(
        magnitude='Magnitude', color='Magnitude', cmap='Viridis',
        width=600, height=600, scale=0.5, colorbar=False
    )
    
    # Manually set contour levels
    contour_levels = np.linspace(potential_values.min(), potential_values.max(), 20)
    
    # Create a contour plot for the potential with enhanced lines and no labels
    contour = hv.operation.contours(hv.Image((R[0], Z[:, 0], potential_values)), levels=contour_levels).opts(
        cmap='Blues', line_width=2, alpha=0.8, colorbar=False, show_legend=False
    )
    
    # Overlay VectorField and Contour
    plot = vector_field * contour
    
    return plot

# Main function to generate plots
def main():
    if MATPLOTLIB_2D_STREAM_CONTOUR:
        matplotlib_2d_stream_contour()
    if MATPLOTLIB_2D_QUIVER:
        matplotlib_2d_quiver()
    if MATPLOTLIB_3D_SURFACE:
        matplotlib_3d_surface()
    if MATPLOTLIB_2D_POTENTIAL_CONTOUR:
        matplotlib_2d_potential_contour()
    if MATPLOTLIB_2D_ELECTRIC_CONTOUR:
        matplotlib_2d_electric_field_contour()
    if PLOTLY_3D_SURFACE:
        plotly_3d_surface()
    if MAYAVI_VECTOR_FIELD:
        mayavi_vector_field()
    if PYVISTA_STREAMLINES:
        pyvista_streamlines()
    if HOLOVIEWS_QUAD_MESH:
        plot = holoviews_vectorfield_plot()
        hv.save(plot, 'quiver_contour_plot.html', backend='bokeh')

if __name__ == "__main__":
    main()