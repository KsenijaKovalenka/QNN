import numpy as np
import matplotlib.pyplot as plt

# classical data
y1 = np.loadtxt("fc_q_c.txt")
y2 = np.loadtxt("fc_c_c.txt")
y3 = np.loadtxt("fc_q_c_deep.txt")
x1 = np.loadtxt("c_batchnumber.txt")

plt.title("Fully connected MNIST")
plt.plot(x1, y1[1:], label="qunatum")
plt.plot(x1, y2[1:], label="classical")
plt.plot(x1, y3[1:], label="quantum deep")
plt.legend()

plt.show()

# qunatum data
x2 = np.loadtxt("q_batchnumber.txt")
y21 = np.loadtxt("fc_q_q.txt")
y22 = np.loadtxt("fc_c_q.txt")
y23 = np.loadtxt("fc_q_q_deep.txt")
y_long = np.loadtxt("fc_q_q_long.txt")
x_long = np.loadtxt("q_batchnumber_new.txt")


# plt.title("Fully connected hoppings")

plt.plot(x2[1:], y21[1:], label="quantum")
plt.plot(x2[1:], y22[1:], label="classical")
plt.legend()
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data
ax.plot(x2[1:], y22[1:], label="classical", color="blueviolet")
ax.plot(x2[1:], y21[1:], label="quantum", color="cornflowerblue")

# Set the axis labels with larger font size
ax.set_xlabel("batch number", fontsize=16)
ax.set_ylabel("cost function", fontsize=16)

# Set the axis ticks in a larger size
ax.tick_params(axis="both", which="major", labelsize=14)

# Add a legend
legend = ax.legend(fontsize=16)

# Show the plot
plt.savefig("cost.png", dpi=300)


# plt.plot(x2[1:], y23[1:], label="deep qunatum")
# plt.plot(x_long[1:], y_long[1:], label="long qunatum")


x3 = np.arange(0, 2401)

plt.title("convolutional hoppings")
y31  = np.loadtxt("cnn_c_q.txt")
y32  = np.loadtxt("cnn_c1_q.txt")
plt.plot(x3[1:], y31[1:], label="qunatum")
plt.plot(x3[1:], y32[1:], label="classical")
plt.legend()
plt.show()

plt.title("convolutional hoppings quantum only")
plt.scatter(x3[1:], y31[1:], s=0.5)
plt.show()

plt.title("convolutional hoppings quantum only run2")
y_conv_2 = np.loadtxt("cnn_c_q_run2.txt")
plt.scatter(x3[1:], y_conv_2[1:], s=0.5)
plt.show()

plt.title("convolutional hoppings quantum only run3")
y_conv_3 = np.loadtxt("cnn_c_q_run3.txt")
plt.scatter(x3[1:], y_conv_3[1:], s=0.5) 
plt.show()

plt.title("convolutional hoppings quantum only run4")
y_conv_4 = np.loadtxt("cnn_c_q_run4.txt")
plt.scatter(x3[1:], y_conv_4[1:], s=0.5)
