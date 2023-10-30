import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
X = np.loadtxt("Averaged homework scores.csv", delimiter=",", dtype=np.float64)
Y = np.loadtxt("Final exam scores.csv", delimiter=",", dtype=np.float64)

# Train test split
X_train, X_test = X[:400], X[400:]
Y_train, Y_test = Y[:400], Y[400:]

# Normalized data
X_mean, X_std = np.mean(X_train), np.std(X_train)
X_train_normalized = (X_train - X_mean) / X_std
X_test_normalized = (X_test - X_mean) / X_std

# Initialize weights and bias
np.random.seed(1)
w = 0
b = 0

# Perform SGD
T = 1000
learning_rate = 0.01
loss_list = []

for iteration in range(T):
    # Select a random data point
    random_index = np.random.randint(len(X_train_normalized))
    x = X_train_normalized[random_index]
    target = Y_train[random_index]

    # Compute the predicted value
    y_pred = w * x + b

    # Compute gradients for weight and bias
    grad_w = 2 * x * (y_pred - target)
    grad_b = 2 * (y_pred - target)

    # Update the weights and bias
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Calculate loss (MSE)
    loss = np.mean((w * X_train_normalized + b - Y_train) ** 2)
    loss_list.append(loss)

# Evaluate the model on the train/test data
y_pred_test = w * X_test_normalized + b
y_pred_train = w * X_train_normalized + b
test_loss = np.mean((y_pred_test - Y_test) ** 2)
train_loss = np.mean((y_pred_train - Y_train) ** 2)

# Plot the testing data and the regression line with custom axis limits
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)
plt.scatter(X_train, Y_train, label="Training Data", color='black', marker = 'x')
x = np.linspace(50, 100, 1000)
x_normalized = (x - X_mean) / X_std
plt.plot(x, w * x_normalized + b, label="Linear Regression", color='red')
plt.xlabel("Averaged homework scores", fontsize=14)
plt.ylabel("Final exam scores", fontsize=14)
plt.legend(fontsize=14)
plt.title('Training MSE=' + str(round(train_loss, 3)), fontsize=14)
plt.xlim(50, 100)
plt.ylim(60, 110)
plt.grid()

plt.subplot(1,2,2)
plt.scatter(X_test, Y_test, label="Testing Data", color='black', marker = 'x')
x = np.linspace(50, 100, 1000)
x_normalized = (x - X_mean) / X_std
plt.plot(x, w * x_normalized + b, label="Linear Regression", color='red')
plt.xlabel("Averaged homework scores", fontsize=14)
plt.ylabel("Final exam scores", fontsize=14)
plt.legend(fontsize=14)
plt.title('Testing MSE=' + str(round(test_loss, 3)), fontsize=14)
plt.xlim(50, 100)
plt.ylim(60, 110)
plt.grid()

plt.savefig('Linear_Regression_SGD.png')
plt.show()

# Monitor Loss v.s Iteration
plt.plot(loss_list[:])
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.title("Loss v.s Iteration")
plt.savefig('Loss_Iteration.png')
plt.show()

# Monitor Loss v.s last 500 Iteration
plt.plot(loss_list[500:])
plt.ylim(10, 13)
plt.xticks([0, 100, 200, 300, 400, 500],[500, 600, 700, 800, 900, 1000])
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("MSE", fontsize=14)
plt.title("Loss v.s Iteration")
plt.savefig('Loss_Iteration.png')
plt.show()

