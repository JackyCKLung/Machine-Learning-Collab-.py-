import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

# Define the data directory
data_dir = r'C:\Users\leona\Downloads\COMP4434_Project\CUB_200_2011\CUB_200_2011\images'

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check the data
for images, labels in dataloader:
    print(images.size(), labels)
    break

# Define the Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv1(x))
        return x * attention

# Define the Part Attention Module
class PartAttention(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(PartAttention, self).__init__()
        self.num_parts = num_parts
        self.conv1 = nn.Conv2d(in_channels, num_parts, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        part_attention = self.softmax(self.conv1(x))
        return part_attention

# Define the Hierarchical Attention Network
class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, num_classes=200, embedding_dim=512, num_parts=15):
        super(HierarchicalAttentionNetwork, self).__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        self.spatial_attention = SpatialAttention(in_channels=2048)
        self.part_attention = PartAttention(in_channels=2048, num_parts=num_parts)

        self.global_fc = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.spatial_fc = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.part_fc = nn.Sequential(
            nn.Linear(2048 * num_parts, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(embedding_dim * 3, num_classes)

    def forward(self, x):
        feature_map = self.backbone(x)
        print("Feature map shape:", feature_map.shape)  # Check the shape again

        # Ensure feature_map has the shape (batch_size, channels, height, width)
        if len(feature_map.shape) == 2:
            feature_map = feature_map.view(feature_map.size(0), 2048, 1, 1)

        global_embedding = self.global_fc(feature_map.mean(dim=[2, 3]))

        spatial_features = self.spatial_attention(feature_map)
        spatial_embedding = self.spatial_fc(spatial_features.mean(dim=[2, 3]))

        part_attention = self.part_attention(feature_map)
        part_features = torch.einsum('bchw,bphw->bcp', feature_map, part_attention)
        print("Part features shape:", part_features.shape)  # Check the shape of part_features

        # Reshape part_features to match the expected input shape for part_fc
        part_features = part_features.view(part_features.size(0), -1)
        print("Reshaped part features shape:", part_features.shape)  # Check the reshaped part_features

        part_embedding = self.part_fc(part_features)

        combined_embedding = torch.cat([global_embedding, spatial_embedding, part_embedding], dim=1)

        logits = self.classifier(combined_embedding)
        return logits, combined_embedding

# Define the CUB200Dataset
class CUB200Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        images_file = os.path.join(root_dir, "images.txt")
        labels_file = os.path.join(root_dir, "image_class_labels.txt")
        split_file = os.path.join(root_dir, "train_test_split.txt")

        self.image_paths = {}
        with open(images_file, 'r') as f:
            for line in f:
                image_id, path = line.strip().split()
                self.image_paths[int(image_id)] = path

        self.labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                image_id, label = line.strip().split()
                self.labels[int(image_id)] = int(label) - 1

        self.split = {}
        with open(split_file, 'r') as f:
            for line in f:
                image_id, is_train = line.strip().split()
                self.split[int(image_id)] = bool(int(is_train))

        self.data = [
            (image_id, self.image_paths[image_id], self.labels[image_id])
            for image_id, is_train in self.split.items()
            if is_train == self.train
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, image_path, label = self.data[idx]
        img_path = os.path.join(self.root_dir, "images", image_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_root = r'C:\Users\leona\Downloads\COMP4434_Project\CUB_200_2011\CUB_200_2011'
    train_dataset = CUB200Dataset(root_dir=dataset_root, train=True, transform=transform)
    test_dataset = CUB200Dataset(root_dir=dataset_root, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalAttentionNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Check if a trained model exists
    model_path = 'trained_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded trained model from 'trained_model.pth'")
    else:
        # Train the model if no trained model exists
        for epoch in range(10):
            print(f"Starting epoch {epoch + 1}")
            model.train()
            epoch_loss = 0
            total_batches = len(train_loader)
            print(f"Total batches in this epoch: {total_batches}")
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"Processing batch {batch_idx + 1}/{total_batches}")
                images, labels = images.to(device), labels.to(device)

                logits, _ = model(images)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}")

        # Save the model after training
        torch.save(model.state_dict(), model_path)
        print("Model saved as 'trained_model.pth'")