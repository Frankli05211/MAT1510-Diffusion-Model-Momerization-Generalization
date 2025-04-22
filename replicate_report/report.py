import matplotlib.pyplot as plt

# Given data pairs
gm_x, gm_y, gm_z = [2, 3, 4, 6], [0.9250, 0.1205, 0.0280, 0.0045], [3700, 482, 112, 18]
gmcifar_x, gmcifar_y, gmcifar_z = [2, 3, 4, 6], [0.9250, 0.1060, 0.0112, 0.0020], [3700, 424, 45, 8]
cifar_x, cifar_y, cifar_z = [2, 3, 4, 6], [0.9250, 0.5630, 0.2162, 0.0180], [3700, 2252, 865, 72]
geometry_element_x, geometry_element_y, geometry_element_z = [2, 3, 4, 6], [0.9250, 0.7917, 0.5783, 0.2730], [3700, 3167, 2313, 1092]
simple_car_x, simple_car_y, simple_car_z = [2, 3, 4, 6], [0.9250, 0.8462, 0.7440, 0.5025], [3700, 3385, 2976, 2010]

# Plotting the data pairs
plt.plot(gm_x, gm_y, label='GM', color='b')
plt.plot(gmcifar_x, gmcifar_y, label='GM Car Combination', color='orange')
plt.plot(cifar_x, cifar_y, label='More CIFAR Car', color='g')
plt.plot(geometry_element_x, geometry_element_y, label='Geometric Element', color='r')
# plt.plot(simple_car_x, simple_car_y, label='Simple Car', color='y')


# Adding labels and title
plt.xlabel('Train Set Size (x1k samples)')
plt.ylabel('Proportion of Replicates')
# plt.title('Line Chart with Multiple Data Pairs')

# Adding a legend
plt.legend()

# Display the plot
plt.savefig("report.png")
plt.show()