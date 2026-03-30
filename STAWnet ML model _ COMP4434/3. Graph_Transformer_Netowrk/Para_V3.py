import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define edge index
edge_index = torch.tensor([[0, 1, 1],
                           [1, 2, 0]], dtype=torch.long).to(device)

# Load the data
train_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\train.npz')
val_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\val.npz')
test_data = np.load(r'C:\Users\leona\Downloads\COMP4434_Project\data\data\METR-LA\test.npz')

# Convert the data to PyTorch tensors and move to device
x_train = torch.tensor(train_data['x'], dtype=torch.float32).to(device)
y_train = torch.tensor(train_data['y'], dtype=torch.float32).to(device)
x_val = torch.tensor(val_data['x'], dtype=torch.float32).to(device)
y_val = torch.tensor(val_data['y'], dtype=torch.float32).to(device)
x_test = torch.tensor(test_data['x'], dtype=torch.float32).to(device)
y_test = torch.tensor(test_data['y'], dtype=torch.float32).to(device)

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
        self.input_projection = nn.Linear(num_nodes * input_dim, hidden_dim)
        
        # Transformer Encoder Layer with dropout
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=0.2, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully Connected Layer for output
        self.fc = nn.Linear(hidden_dim, output_dim * num_nodes * seq_len)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, temporal_data):
        # GNN forward pass
        x = self.gnn(x, edge_index)
        x = F.relu(x)

        # Reshape temporal_data to match expected input for Transformer
        batch_size, seq_len, num_nodes, input_dim = temporal_data.shape
        temporal_data = temporal_data.view(batch_size, seq_len, num_nodes * input_dim)
        temporal_data = self.input_projection(temporal_data)

        # Transformer forward pass
        transformer_output = self.transformer(temporal_data)
        last_time_step_output = transformer_output[:, -1, :]

        # Ensure x has the same shape as last_time_step_output
        x = x.view(batch_size, -1)
        x = x[:, :last_time_step_output.size(1)]

        # Combine GNN and Transformer outputs
        combined = x + last_time_step_output

        # Apply dropout
        combined = self.dropout(combined)

        # Output layer
        out = self.fc(combined).view(batch_size, seq_len, num_nodes, -1)

        return out

num_nodes = 207
seq_len = 12
input_dim = 1
hidden_dim = 128  
num_heads = 8
num_layers = 4
output_dim = 1

# Example training loop
model = GraphTransformerNetwork(num_nodes, seq_len, input_dim, hidden_dim, num_heads, num_layers, output_dim).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

print(x_train.shape)

for epoch in range(100):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch[:, -1], edge_index, x_batch)
        loss = criterion(output.view(-1), y_batch.view(-1))
        loss.backward()
        optimizer.step()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch[:, -1], edge_index, x_batch)
            val_loss += criterion(output.view(-1), y_batch.view(-1)).item()
    
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}')
    scheduler.step(val_loss / len(val_loader))

# Test the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch[:, -1], edge_index, x_batch)
        test_loss += criterion(output.view(-1), y_batch.view(-1)).item()

print(f'Test Loss: {test_loss / len(test_loader)}')