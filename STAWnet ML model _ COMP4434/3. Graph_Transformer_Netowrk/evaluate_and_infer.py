import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
from Testing import HierarchicalAttentionNetwork, CUB200Dataset  # Adjust the import as needed

# Define the data directory and transformations
data_dir = r'C:\Users\leona\Downloads\COMP4434 Project\CUB_200_2011\CUB_200_2011\images'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
dataset_root = r'C:\Users\leona\Downloads\COMP4434_Project\CUB_200_2011\CUB_200_2011'
test_dataset = CUB200Dataset(root_dir=dataset_root, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalAttentionNetwork().to(device)
model.load_state_dict(torch.load('trained_model.pth'))  # Load your trained model
model.eval()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# Evaluate the model on the test dataset
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Function to preprocess and predict a single image
def predict_image(image_path, model, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(image_tensor)
        _, predicted = torch.max(logits, 1)
    
    return predicted.item(), image

# Function to process all images in all subfolders
def predict_all_images_in_folder(root_folder, model, transform, device):
    predictions = {}
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(subdir, file)
                predicted_class, image = predict_image(image_path, model, transform, device)
                predictions[image_path] = predicted_class
                print(f"Predicted class for {image_path}: {predicted_class}")
                # Display the image and prediction for one example
                plt.imshow(image)
                plt.title(f"Predicted class: {predicted_class}")
                plt.axis('off')
                plt.show()
                return predictions

# Example usage
root_folder = r'C:\Users\leona\Downloads\COMP4434_Project\CUB_200_2011\CUB_200_2011\images'  # Replace with your root folder
predictions = predict_all_images_in_folder(root_folder, model, transform, device)