import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Number of devices
num_devices = 100

# Range for X and Y values
x_range = (0, 215)
y_range = (0, 215)


# Generate random X and Y values within the specified range
x_values = np.random.uniform(x_range[0], x_range[1], num_devices)
y_values = np.random.uniform(y_range[0], y_range[1], num_devices)

# Dataset generation
dataset = []

for i in range(100):
    dataset.append(("D"+str(i+1), round(x_values[i], 2), round(y_values[i], 2)))


# Create the corresponding file
filename = "device_coordinates.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Device Number', 'X Value', 'Y Value'])
    for row in dataset:
        writer.writerow(row)
file.close()



# Create a DataFrame from the provided data
dataframe = pd.DataFrame(dataset, columns=['Device Number', 'X Value', 'Y Value'])


# Plot a scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(dataframe['X Value'], dataframe['Y Value'], c='blue', label='Data Points')

# Add labels and title
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.title('Scatter Plot of Given Coordinates')

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
