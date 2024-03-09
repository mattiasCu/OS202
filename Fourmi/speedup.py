import matplotlib.pyplot as plt

# Assuming the x-axis represents the number of compute processors (+1 for display)
# and the y-axis represents the speedup.
# The exact values are not given, so this will use a general trend similar to the image.

# Sample data (x: number of processors, y: speedup)
processors = [2, 4, 6, 8, 10, 12, 14]
speedup = [4.8, 6.9, 6.5, 6.32, 6.44, 5.83, 6.39]  # Example speedup values based on the image trend

plt.plot(processors, speedup, 'r-', marker='o')  # 'b-' is for a blue line
plt.title('Speed up')
plt.xlabel('Nombre de processeurs de calcul ')
plt.ylabel('Speed up')
plt.grid(True)
plt.xticks(processors)  # Set x-ticks to be exactly as the number of processors
plt.yticks(range(0, int(max(speedup)) + 2))  # Assuming the speed up goes from 0 to max + 2 for visual clarity
plt.tight_layout()  # Adjust the padding of the plot
plt.show()
