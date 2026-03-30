import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from your_model import SimpleEmbeddingModel  # Adjust the import based on your structure
from your_util import load_dataset  # Adjust the import based on your structure

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
data_path = '.'  # Adjust path as needed
batch_size = 64
dataloader = load_dataset(data_path, batch_size, batch_size, batch_size)
train_loader = dataloader['train_loader']
test_loader = dataloader['test_loader']

# Initialize model
model = SimpleEmbeddingModel(in_channels=1)  # Adjust in_channels if necessary
model.to(device)

# Training parameters
num_epochs = 100
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

# Initialize scaler
scaler = StandardScaler()

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        output = model(x)

        # Fit the scaler on the training labels
        if epoch == 0:  # Fit only once
            scaler.fit(y.cpu().numpy().reshape(-1, y.shape[-1]))

        # Calculate loss
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

# Testing loop
print("Starting testing...")
model.eval()
outputs = []
realy = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        preds = model(x)

        # Store predictions and real values
        outputs.append(preds.cpu())
        realy.append(y.cpu())

# Concatenate outputs
yhat = torch.cat(outputs, dim=0)
realy = torch.cat(realy, dim=0)

# Calculate metrics
mae, mape, rmse = [], [], []
for i in range(12):
    pred = scaler.inverse_transform(yhat[:, i, :, :].squeeze().numpy())
    real = realy[:, i, :, :].squeeze().numpy()

    # Calculate metrics here (implement your metric calculations)
    # Example:
    mae.append(np.mean(np.abs(pred - real)))
    mape.append(np.mean(np.abs((pred - real) / real)) * 100)
    rmse.append(np.sqrt(np.mean((pred - real) ** 2)))

# Print overall metrics
print("Overall Metrics:")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}") 