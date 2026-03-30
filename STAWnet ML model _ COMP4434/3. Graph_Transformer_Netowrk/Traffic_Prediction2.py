import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # Ensure you have PyTorch Geometric installed
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Define edge index
edge_index = torch.tensor([[0, 1, 1],
                           [1, 2, 0]], dtype=torch.long)

# Load the data
train_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\train.npz')
val_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\val.npz')
test_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\test.npz')

# Convert the data to PyTorch tensors
x_train = torch.tensor(train_data['x'], dtype=torch.float32)
y_train = torch.tensor(train_data['y'], dtype=torch.float32)
x_val = torch.tensor(val_data['x'], dtype=torch.float32)
y_val = torch.tensor(val_data['y'], dtype=torch.float32)
x_test = torch.tensor(test_data['x'], dtype=torch.float32)
y_test = torch.tensor(test_data['y'], dtype=torch.float32)

# Optionally, you can also load offsets if needed
x_offsets = train_data['x_offsets']
y_offsets = train_data['y_offsets']

from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create datasets
train_dataset = TrafficDataset(x_train, y_train)
val_dataset = TrafficDataset(x_val, y_val)
test_dataset = TrafficDataset(x_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class GraphTransformerNetwork(nn.Module):
    def __init__(self, num_nodes, seq_len, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(GraphTransformerNetwork, self).__init__()
        
        # GNN Layer
        self.gnn = GCNConv(input_dim, hidden_dim)
        
        # Input projection for Transformer
        self.input_projection = nn.Linear(num_nodes * input_dim, hidden_dim)  # From 207 to hidden_dim
        
        # Transformer Encoder Layer
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully Connected Layer for output
        self.fc = nn.Linear(hidden_dim, output_dim * num_nodes * seq_len)  # Adjusted to match the target size
        
    def forward(self, x, edge_index, temporal_data):
        # x: Node features
        # edge_index: Graph connectivity
        # temporal_data: Sequential data for Transformer

        # GNN forward pass
        x = self.gnn(x, edge_index)
        x = F.relu(x)

        # Reshape temporal_data to match expected input for Transformer
        # temporal_data has shape (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, input_dim = temporal_data.shape

        # Flatten the node and feature dimensions for projection
        temporal_data = temporal_data.view(batch_size, seq_len, num_nodes * input_dim)  # Shape: (batch_size, seq_len, 207)

        # Project input to match Transformer input dimension
        temporal_data = self.input_projection(temporal_data)  # Shape: (batch_size, seq_len, hidden_dim)

        # Transformer forward pass
        transformer_output = self.transformer(temporal_data)  # Shape: (batch_size, seq_len, hidden_dim)

        # Use the last time step's output
        last_time_step_output = transformer_output[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Ensure x has the same shape as last_time_step_output
        x = x.view(batch_size, -1)  # Flatten x to (batch_size, num_nodes * hidden_dim)
        x = x[:, :last_time_step_output.size(1)]  # Ensure x matches the hidden_dim size

        # Combine GNN and Transformer outputs
        combined = x + last_time_step_output

        # Output layer
        out = self.fc(combined).view(batch_size, seq_len, num_nodes, -1)  # Reshape to match the target size

        return out

num_nodes = 207  # Example number of nodes
seq_len = 12     # Example sequence length
input_dim = 1   # Example input feature dimension
hidden_dim = 32  # Hidden dimension for both GNN and Transformer
num_heads = 4    # Number of attention heads in Transformer
num_layers = 2   # Number of Transformer layers
output_dim = 1   # Output dimension (e.g., traffic speed)

# Example training loop
model = GraphTransformerNetwork(num_nodes, seq_len, input_dim, hidden_dim, num_heads, num_layers, output_dim)
criterion = torch.nn.MSELoss()  # Example loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(x_train.shape)  # Should be (num_samples, seq_len, num_nodes, input_dim)

for epoch in range(10):  # Example number of epochs
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch[:, -1], edge_index, x_batch)  # Use the last time step's node features for GNN input and full sequence for Transformer input
        loss = criterion(output.view(-1), y_batch.view(-1))  # Flatten both output and target to match sizes for MSE loss calculation
        loss.backward()
        optimizer.step()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x_batch, y_batch in val_loader:
            output = model(x_batch[:, -1], edge_index, x_batch)
            val_loss += criterion(output.view(-1), y_batch.view(-1)).item()  # Flatten both output and target to match sizes for MSE loss calculation
    
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}')


# Test the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for x_batch, y_batch in test_loader:
        output = model(x_batch[:, -1], edge_index, x_batch)
        test_loss += criterion(output.view(-1), y_batch.view(-1)).item()  # Flatten both output and target to match sizes for MSE loss calculation

print(f'Test Loss: {test_loss / len(test_loader)}')
    