import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load Data
train_data = np.load('train.npz')
val_data = np.load('val.npz')
test_data = np.load('test.npz')

# Extract the data
train_x, train_y = train_data['x'], train_data['y']
val_x, val_y = val_data['x'], val_data['y']
test_x, test_y = test_data['x'], test_data['y']

# Step 2: Flatten the Input Data
train_x_flat = train_x.reshape(train_x.shape[0], -1)
val_x_flat = val_x.reshape(val_x.shape[0], -1)
test_x_flat = test_x.reshape(test_x.shape[0], -1)

# Step 3: Prepare the Target Data for All Nodes
train_y_flat = train_y[:, -1, :, 0].reshape(train_y.shape[0], -1)  # Shape: (samples, nodes)
val_y_flat = val_y[:, -1, :, 0].reshape(val_y.shape[0], -1)
test_y_flat = test_y[:, -1, :, 0].reshape(test_y.shape[0], -1)

# Step 4: Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_x_flat)
X_val_scaled = scaler.transform(val_x_flat)
X_test_scaled = scaler.transform(test_x_flat)

# Function to Build and Train CNN for Each Node
def train_cnn_for_node(node_index):
    print(f"\n--- Training CNN for Node {node_index} ---")
    
    # Get target values for the current node
    train_y_node = train_y_flat[:, node_index]
    val_y_node = val_y_flat[:, node_index]
    test_y_node = test_y_flat[:, node_index]

    # Build CNN Model
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))  # Convolution layer
    cnn_model.add(MaxPooling1D(pool_size=2))  # Max pooling layer
    cnn_model.add(Flatten())  # Flatten layer to convert 2D features to 1D
    cnn_model.add(Dense(1))  # Output layer

    # Compile the model
    cnn_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the CNN Model
    cnn_history = cnn_model.fit(
        X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1), train_y_node,
        epochs=10, batch_size=32,
        validation_data=(X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1), val_y_node),
        verbose=0
    )

    # Evaluate Test Performance
    y_pred_cnn = cnn_model.predict(X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1))
    mae_cnn = mean_absolute_error(test_y_node, y_pred_cnn)
    rmse_cnn = np.sqrt(mean_squared_error(test_y_node, y_pred_cnn))
    print(f"CNN Test MAE: {mae_cnn:.4f}")
    print(f"CNN Test RMSE: {rmse_cnn:.4f}")

    # Plot Predictions vs Actual Values
    plt.figure(figsize=(10, 6))
    plt.plot(test_y_node[:100], label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred_cnn[:100], label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.title(f"CNN Predictions vs Actual Traffic (Node {node_index})")
    plt.xlabel("Sample Index")
    plt.ylabel("Traffic Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Train the CNN for Multiple Nodes
num_nodes = train_y_flat.shape[1]  # Total number of nodes
for node_index in range(3):  # Train for the first 3 nodes
    train_cnn_for_node(node_index)