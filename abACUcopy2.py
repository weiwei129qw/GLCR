import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define the original x, y, and z data points.
x = np.array([-5, -4, -3, -2, -1])
y = np.array([-5, -4, -3, -2, -1])
z = np.array([
    1.198533734, 1.198737709, 1.224306324, 3.130448208, 3.15,
    11.00397703, 1.198608102, 1.198801031, 1.22438071, 1.23,
    3.130522594, 11.00405142, 1.199352519, 1.19955395, 1.21,
    1.225125042, 3.131266927, 11.00479575, 1.206795428, 1.95,
    1.206978212, 1.232567936, 3.138709821, 11.01223864, 3.14
]).reshape(5, 5)  # Reshape z to a 5x5 matrix

# Create grid coordinates for the output of the griddata function.
xi = np.linspace(-5, -1, 40)
yi = np.linspace(-5, -1, 40)
xi, yi = np.meshgrid(xi, yi)

# Grid the data using cubic interpolation.
zi = griddata((x.repeat(5), y.repeat(5)), z.flatten(), (xi, yi), method='cubic')

# Plotting the 3D figure using matplotlib.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Log scale for y axis
ax.set_yscale('log')

# Create the surface plot and set the scale to logarithmic for the Y axis.
surf = ax.plot_surface(xi, yi, zi, cmap='viridis')

# Set labels and title with logarithmic notation
ax.set_xlabel('log(a)')
ax.set_ylabel('log(Δ)')
ax.set_zlabel('MSE')
ax.set_title('TR20')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
