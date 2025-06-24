import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dataset class for loading preprocessed EEG segments
class EEGDataset(Dataset):
    def __init__(self, data_path):
        full_data = np.load(data_path / 'preprocessed_data.npy')
        raw_labels = np.load(data_path / 'labels.npy')

        print("Loaded preprocessed shape:", full_data.shape)
        print("Labels shape:", raw_labels.shape)
        print("Dtype:", full_data.dtype)

        self.data = torch.tensor(full_data, dtype=torch.float32)

        # Convert string class names to numerical indices for training
        self.class_names = sorted(set(raw_labels))  # Ensures consistent label ordering
        self.class_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        self.index_to_class = {idx: label for label, idx in self.class_to_index.items()}

        # Map each string label to its corresponding integer index
        numeric_labels = [self.class_to_index[label] for label in raw_labels]
        self.labels = torch.tensor(numeric_labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# A simple 1D CNN designed to classify EEG time-series data
class SimpleEEG1DCNN(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SimpleEEG1DCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.LazyLinear(128)  # Automatically infers input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)

# Training loop including confusion matrix evaluation after final epoch
def train_model(data_folder, num_epochs=30, batch_size=32, learning_rate=0.001):
    dataset = EEGDataset(data_folder)
    num_classes = len(dataset.class_names)
    input_channels = dataset.data.shape[1]

    # 80/20 split for training and validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = SimpleEEG1DCNN(in_channels=input_channels, n_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Val Acc: {accuracy:.2f}%")

    # After training, compute and show confusion matrix for final validation
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Validation Confusion Matrix (Final Epoch)")
    plt.tight_layout()
    plt.show()

    model_path = data_folder / 'eeg_1dcnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python torch_baseline_model_gen.py <recording_folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]

    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_path = base_dir / 'data' / 'processed' / 'training' / folder_name
    model_output_path = base_dir / 'data' / 'model' / folder_name

    if not processed_path.exists():
        print(f"Error: folder '{processed_path}' does not exist.")
        sys.exit(1)

    os.makedirs(model_output_path, exist_ok=True)

    train_model(processed_path)

    model_file = processed_path / 'eeg_1dcnn_model.pth'
    destination_file = model_output_path / 'eeg_1dcnn_model.pth'
    os.replace(model_file, destination_file)

    print(f"Model moved to {destination_file}")
