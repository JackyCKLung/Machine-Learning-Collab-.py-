import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler  # Or MinMaxScaler
from STAWnet import util  # Import util from the same package

# Define batch_size here
batch_size = 64 

test_data = np.load('test.npz')  
print(test_data.files)  # Print the available keys in the archive

# Prepare the test dataset
test_images = test_data['x'].transpose(0, 3, 1, 2)  # Assuming 'x' contains the image data
test_labels = test_data['y']  # Assuming 'y' contains the labels

test_dataset = TensorDataset(torch.from_numpy(test_images).float(), torch.from_numpy(test_labels).long())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

# Assuming engine is defined as follows, based on your input:
class Engine:  # Placeholder for the Engine class
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, learning_rate, weight_decay, device, gat_bool, addaptadj, adjinit, aptonly, emb_length, noapt):
        self.model = SimpleEmbeddingModel() # Assuming SimpleEmbeddingModel is defined earlier
# Define Simple Embedding Model
class SimpleEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=128, in_channels=1):
        super(SimpleEmbeddingModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 
                              kernel_size=(3, 3), 
                              padding='same',
                              stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((12, 207))  # Match required dimensions
        
        # Calculate final output size
        conv_out_channels = 32
        self.fc = nn.Linear(conv_out_channels * 12 * 207, 12 * 207)  # Output size matches required dimensions

    def forward(self, x):
        print("Input shape:", x.shape)
        
        x = self.conv1(x)
        print("After Conv1:", x.shape)
        
        x = self.relu(x)
        print("After ReLU:", x.shape)
        
        x = self.adaptive_pool(x)
        print("After AdaptivePool:", x.shape)
        
        x = x.view(x.size(0), -1)
        print("After Flatten:", x.shape)
        
        x = self.fc(x)
        print("Final output:", x.shape)
        return x

scaler = StandardScaler()  # Or MinMaxScaler()
device = torch.device('cpu')
engine = Engine(scaler=None,  # Replace with your actual scaler
                 in_dim=None,  # Replace with your actual args.in_dim
                 seq_length=None,  # Replace with your actual args.seq_length
                 num_nodes=None,  # Replace with your actual args.num_nodes
                 nhid=None,  # Replace with your actual args.nhid
                 dropout=None,  # Replace with your actual args.dropout
                 learning_rate=None,  # Replace with your actual args.learning_rate
                 weight_decay=None,  # Replace with your actual args.weight_decay
                 device=device,  
                 gat_bool=None,  # Replace with your actual args.gat_bool
                 addaptadj=None,  # Replace with your actual args.addaptadj
                 adjinit=None,  
                 aptonly=None,  # Replace with your actual args.aptonly
                 emb_length=None,  # Replace with your actual args.emb_length
                 noapt=None)  # Replace with your actual args.noapt

engine.model = SimpleEmbeddingModel(
    embedding_size=128,
    in_channels=1  # Your input has 1 channel
)

# Define your test function
def test(engine, data_loader, device, scaler):
    """Test the model"""
    outputs = []
    realy = []

    # First, collect all data to fit the scaler
    all_data = []
    with torch.no_grad():
        for x, y in data_loader:
            all_data.append(y.numpy())
    all_data = np.concatenate(all_data, axis=0)
    
    # Fit the scaler
    if scaler is None:
        scaler = StandardScaler()
    scaler.fit(all_data.reshape(-1, all_data.shape[-1]))

    # Now proceed with testing
    with torch.no_grad():
        for x, y in data_loader:
            print("Before processing:")
            print(f"x shape: {x.shape}")  # [64, 1, 12, 207]
            print(f"y shape: {y.shape}")  # [64, 12, 207, 1]
            
            testx = x.float().to(device)
            testy = y.float().to(device)
            
            print("After preprocessing:")
            print(f"testx shape: {testx.shape}")
            print(f"testy shape: {testy.shape}")
            
            # Forward pass
            preds = engine.model(testx)
            print("Model output shape:", preds.shape)
            
            # Reshape predictions to match target shape
            preds = preds.view(preds.shape[0], 12, 207, 1)
            print("Reshaped predictions:", preds.shape)
            
            outputs.append(preds)
            realy.append(testy)

    # Concatenate batches
    yhat = torch.cat(outputs, dim=0)
    realy = torch.cat(realy, dim=0)

    print("Final shapes:")
    print(f"yhat shape: {yhat.shape}")
    print(f"realy shape: {realy.shape}")

    # Calculate metrics
    mae, mape, rmse = [], [], []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,i,:,:].squeeze().cpu().numpy())
        real = realy[:,i,:,:].squeeze().cpu().numpy()
        
        # Convert back to tensor for metric calculation
        pred = torch.FloatTensor(pred).to(device)
        real = torch.FloatTensor(real).to(device)
        
        metrics = util.metric(pred, real)
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
        print(f'Horizon {i+1}, MAE: {metrics[0]:.4f}, MAPE: {metrics[1]:.4f}, RMSE: {metrics[2]:.4f}')

    return {'mae': mae, 'mape': mape, 'rmse': rmse}

# Function to calculate metrics
def forward(self, x):
    x = self.relu(self.conv1(x))
    print("After Conv:", x.shape)  # Check shape after Conv
    x = self.maxpool(x)
    print("After MaxPool:", x.shape)  # Check shape after MaxPool
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

# Masked metrics functions
def masked_mae(preds, labels, null_val=np.nan):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mse(preds, labels, null_val=np.nan):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    return torch.mean(((preds - labels) ** 2) * mask)

# Verify metrics function
def verify_metrics(metrics, horizon):
    mae, mape, rmse = metrics
    
    # Expected ranges vary by horizon
    if horizon == 0:  # First horizon
        assert 1.8 <= mae <= 2.5, "MAE out of expected range"
        assert 0.07 <= mape <= 0.09, "MAPE out of expected range"
        assert 3.8 <= rmse <= 4.5, "RMSE out of expected range"
    
    elif horizon == 11:  # Last horizon
        assert 3.0 <= mae <= 4.0, "MAE out of expected range"
        assert 0.12 <= mape <= 0.15, "MAPE out of expected range"
        assert 5.0 <= rmse <= 6.0, "RMSE out of expected range"

# Example usage
print("Starting testing...")
mae_results, mape_results, rmse_results = test(engine, test_loader, device, scaler)

for i in range(12):
    metrics = (mae_results[i], mape_results[i], rmse_results[i])
    print(f"Horizon {i + 1}, MAE: {metrics[0]:.4f}, MAPE: {metrics[1]:.4f}, RMSE: {metrics[2]:.4f}")
    
    # Verify metrics for the current horizon
    verify_metrics(metrics, i)

# Make sure to export the class
__all__ = ['SimpleEmbeddingModel']

state_dict = torch.load(checkpoint_path)
print("Keys in the saved state_dict:", state_dict.keys())