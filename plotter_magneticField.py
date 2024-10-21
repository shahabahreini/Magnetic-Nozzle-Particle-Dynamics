import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
B0 = 1  # Arbitrary unit for magnetic field strength
r0 = 1  # Arbitrary unit for reference radius

# Create a 2D grid
R, Z = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))

# Calculate the magnetic field components for 2D
r = np.sqrt(R**2 + Z**2)
BR = B0 * r0**2 * R / (R**2 + Z**2)**(3/2)
BZ = B0 * r0**2 * Z / (R**2 + Z**2)**(3/2)

# Calculate the magnitude of the magnetic field
B_mag = np.sqrt(BR**2 + BZ**2)

# Create the 2D plot
plt.figure(figsize=(12, 10), dpi=80)
plt.quiver(R, Z, BR, BZ, B_mag, cmap='viridis', pivot='mid', scale=10)
cbar = plt.colorbar(label='Magnetic Field Strength')
cbar.ax.tick_params(labelsize=12)
plt.title('2D Magnetic Field Components in Cylindrical Coordinates', fontsize=16)
plt.xlabel('R (Radial Distance)', fontsize=14)
plt.ylabel('z (Axial Distance)', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.axis('equal')
plt.tight_layout()

# Save the 2D plot as a high-quality PNG file
plt.savefig('agnetic_field_plot_2d.png', dpi=300, bbox_inches='tight')

# Display the 2D plot
plt.show()

# Create a 3D grid
X, Y, Z = np.meshgrid(np.linspace(-2, 2, 6), np.linspace(-2, 2, 10), np.linspace(-2, 2, 6))

# Calculate the magnetic field components for 3D
r = np.sqrt(X**2 + Y**2 + Z**2)
BX = B0 * r0**2 * X / r**5
BY = B0 * r0**2 * Y / r**5
BZ = B0 * r0**2 * Z / r**5

# Calculate the magnitude of the magnetic field
B_mag = np.sqrt(BX**2 + BY**2 + BZ**2)

# Create the 3D plot
fig = plt.figure(figsize=(12, 10), dpi=80)
ax = fig.add_subplot(111, projection='3d')

# Plot the vector field
norm = plt.Normalize(vmin=0, vmax=B_mag.max())
colors = plt.cm.viridis(norm(B_mag))
ax.quiver(X, Y, Z, BX, BY, BZ, length=0.5, normalize=True, color=colors.reshape(-1, 4))

ax.set_title('3D Magnetic Field Vector Field', fontsize=16)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('Y', fontsize=14)
ax.set_zlabel('Z', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Magnetic Field Strength')
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()

# Save the 3D plot as a high-quality PNG file
plt.savefig('magnetic_field_plot_3d.png', dpi=300, bbox_inches='tight')

# Display the 3D plot
plt.show()

print("The 2D plot has been saved as 'magnetic_field_plot_2d.png' in the /mnt/data directory.")
print("The 3D plot has been saved as 'magnetic_field_plot_3d.png' in the /mnt/data directory.")
