import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the Data
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

# Step 3: Prepare the Target Data
train_y_flat = train_y[:, -1, :, 0].reshape(train_y.shape[0], -1)
val_y_flat = val_y[:, -1, :, 0].reshape(val_y.shape[0], -1)
test_y_flat = test_y[:, -1, :, 0].reshape(test_y.shape[0], -1)

node_index = 0
train_y_svm = train_y_flat[:, node_index]
val_y_svm = val_y_flat[:, node_index]
test_y_svm = test_y_flat[:, node_index]

# Step 4: Standardize the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_x_flat)
X_val_scaled = scaler.transform(val_x_flat)
X_test_scaled = scaler.transform(test_x_flat)

# Step 5: Train the SVM Model
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train_scaled, train_y_svm)

# Evaluate Training and Validation Scores
train_score = svm_model.score(X_train_scaled, train_y_svm)
val_score = svm_model.score(X_val_scaled, val_y_svm)
print(f"Training R^2 Score: {train_score:.4f}")
print(f"Validation R^2 Score: {val_score:.4f}")

# Step 6: Predict on Test Set
y_pred = svm_model.predict(X_test_scaled)

# Step 7: Evaluate Test Performance
mae = mean_absolute_error(test_y_svm, y_pred)
rmse = np.sqrt(mean_squared_error(test_y_svm, y_pred))
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# Step 8: Visualize Predictions
plt.figure(figsize=(10, 6))
plt.plot(test_y_svm[:100], label='Actual', marker='o')
plt.plot(y_pred[:100], label='Predicted', marker='x')
plt.legend()
plt.title("SVM Predictions vs Actual Traffic (Node 0)")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Value")
plt.show()