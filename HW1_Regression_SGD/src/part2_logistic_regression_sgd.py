import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load the dataset
X_homework = np.loadtxt("Averaged homework scores.csv", delimiter=",", dtype=np.float64)
X_final_exam = np.loadtxt("Final exam scores.csv", delimiter=",", dtype=np.float64)
y = np.loadtxt("Results.csv", delimiter=",", dtype=bool)

# Merge the features into one dataset
X = np.column_stack((X_homework, X_final_exam))

# Train-test split
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Compute mean and standard deviation using the training data
X_train_mean, X_train_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)

# Normalize the dataset
X_train_normalized = (X_train - X_train_mean) / X_train_std
X_test_normalized = (X_test - X_train_mean) / X_train_std

# Append a column of 1s for the bias term
X_train_normalized = np.column_stack((np.ones(len(X_train_normalized)), X_train_normalized))
X_test_normalized = np.column_stack((np.ones(len(X_test_normalized)), X_test_normalized))

# Initialize weights
np.random.seed(0)
w = np.random.rand(X_train_normalized.shape[1])  # Weights

# Perform SGD
T = 1000
learning_rate = 0.75
loss_list = []
for iteration in range(T):
    # Select a random data point
    random_index = np.random.randint(len(X_train_normalized))
    x = X_train_normalized[random_index]
    target = y_train[random_index]

    # Compute the predicted values
    y_pred = sigmoid(np.dot(x, w))

    # Compute gradients for weights
    grad_w = np.dot(x.T, (y_pred - target))

    # Update the weights
    w -= learning_rate * grad_w

    y_pred_train = sigmoid(np.dot(X_train_normalized, w))
    loss = -np.mean(y_train * np.log(y_pred_train) + (1 - y_train) * np.log(1 - y_pred_train))
    loss_list.append(loss)


# Apply the predictor to the testing data
y_pred_test = sigmoid(np.dot(X_test_normalized, w))
y_pred_train = sigmoid(np.dot(X_train_normalized, w))

# Make hard decisions based on the predicted probability
y_pred_test_binary = np.where(y_pred_test >= 0.5, 1, 0)
y_pred_train_binary = np.where(y_pred_train >= 0.5, 1, 0)

# Compute the logistic loss
logistic_test_loss = -np.mean(y_test * np.log(y_pred_test) + (1 - y_test) * np.log(1 - y_pred_test))
logistic_train_loss = -np.mean(y_train * np.log(y_pred_train) + (1 - y_train) * np.log(1 - y_pred_train))

# Plotting the figure
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
blue_label = False
red_label = False
for i in range(len(X_train)):
    if y_train[i] == 1:
        if blue_label == False:
            plt.scatter(X_train_normalized[i, 1], X_train_normalized[i, 2], color='blue', marker = 'x', label='Admitted')
            blue_label = True
        else:
            plt.scatter(X_train_normalized[i, 1], X_train_normalized[i, 2], color='blue', marker = 'x')
    else:
        if red_label == False:
            plt.scatter(X_train_normalized[i, 1], X_train_normalized[i, 2], color='red', marker = 'x', label='Rejected')
            red_label = True
        else:
            plt.scatter(X_train_normalized[i, 1], X_train_normalized[i, 2], color='red', marker = 'x')

# Plot the decision boundary
x_values = np.array([np.min(X_test_normalized[:, 1]), np.max(X_test_normalized[:, 1])])
y_values = -(w[0] + w[1] * x_values) / w[2]
plt.plot(x_values, y_values, label='Decision Boundary', color = 'k')

# Adjust the axis limits
plt.xlim(np.min(X_test_normalized[:, 1]) - 0.5, np.max(X_test_normalized[:, 1]) + 0.5)
plt.ylim(np.min(X_test_normalized[:, 2]) - 0.5, np.max(X_test_normalized[:, 2]) + 0.5)

plt.title("Training Logistic Loss=" + str(round(logistic_train_loss, 3)), fontsize=14)
plt.xlabel("Averaged homework scores", fontsize=14)
plt.ylabel("Final exam scores", fontsize=14)
plt.legend()
plt.grid()

plt.subplot(1,2,2)
blue_label = False
red_label = False
for i in range(len(X_test)):
    if y_test[i] == 1:
        if blue_label == False:
            plt.scatter(X_test_normalized[i, 1], X_test_normalized[i, 2], color='blue', marker = 'x', label='Admitted')
            blue_label = True
        else:
            plt.scatter(X_test_normalized[i, 1], X_test_normalized[i, 2], color='blue', marker = 'x')
    else:
        if red_label == False:
            plt.scatter(X_test_normalized[i, 1], X_test_normalized[i, 2], color='red', marker = 'x', label='Rejected')
            red_label = True
        else:
            plt.scatter(X_test_normalized[i, 1], X_test_normalized[i, 2], color='red', marker = 'x')

# Plot the decision boundary
x_values = np.array([np.min(X_test_normalized[:, 1]), np.max(X_test_normalized[:, 1])])
y_values = -(w[0] + w[1] * x_values) / w[2]
plt.plot(x_values, y_values, label='Decision Boundary', color = 'k')

# Adjust the axis limits
plt.xlim(np.min(X_test_normalized[:, 1]) - 0.5, np.max(X_test_normalized[:, 1]) + 0.5)
plt.ylim(np.min(X_test_normalized[:, 2]) - 0.5, np.max(X_test_normalized[:, 2]) + 0.5)

plt.title("Testing Logistic Loss=" + str(round(logistic_test_loss, 3)), fontsize=14)
plt.xlabel("Averaged homework scores", fontsize=14)
plt.ylabel("Final exam scores", fontsize=14)
plt.legend()
plt.grid()
plt.show()

plt.plot(loss_list)
plt.ylim(0.3 , 0.8)
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Logistic Loss", fontsize=14)
plt.title("Loss v.s Iteration")
plt.savefig('Loss_Iteration.png')
plt.show()

