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

# Loads EEG segment data and converts string labels to numeric format
class EEGDataset(Dataset):
    def __init__(self, data_directory):
        eeg_data = np.load(data_directory / 'preprocessed_data.npy')
        label_strings = np.load(data_directory / 'labels.npy')

        print("Loaded preprocessed shape:", eeg_data.shape)
        print("Labels shape:", label_strings.shape)
        print("Dtype:", eeg_data.dtype)

        self.samples = torch.tensor(eeg_data, dtype=torch.float32)

        self.class_names = sorted(set(label_strings))
        self.class_to_index = {label: idx for idx, label in enumerate(self.class_names)}
        self.index_to_class = {idx: label for label, idx in self.class_to_index.items()}

        numeric_labels = [self.class_to_index[label] for label in label_strings]
        self.labels = torch.tensor(numeric_labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index].transpose(0, 1)  # (1000, 8) → (8, 1000)
        return sample, self.labels[index]


# A basic 1D CNN model designed for EEG sequence classification
class EEGClassifier1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EEGClassifier1DCNN, self).__init__()

        self.conv_layer_1 = nn.Conv1d(input_channels, 8, kernel_size=5, padding=2)
        self.batch_norm_1 = nn.BatchNorm1d(8)
        self.activation_1 = nn.ReLU()
        self.max_pool_1 = nn.MaxPool1d(kernel_size=2)

        self.conv_layer_2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm1d(16)
        self.activation_2 = nn.ReLU()
        self.max_pool_2 = nn.MaxPool1d(kernel_size=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Collapse time dimension
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(16, num_classes)

    def forward(self, input_sequence):
        output = self.conv_layer_1(input_sequence)
        output = self.batch_norm_1(output)
        output = self.activation_1(output)
        output = self.max_pool_1(output)

        output = self.conv_layer_2(output)
        output = self.batch_norm_2(output)
        output = self.activation_2(output)
        output = self.max_pool_2(output)

        output = self.global_avg_pool(output)   
        output = output.squeeze(-1)             
        output = self.dropout(output)
        predictions = self.output_layer(output)
        return predictions

# Trains the 1D CNN model on EEG data and evaluates on a validation split
def train_eeg_cnn_model(processed_data_dir, num_epochs=20, batch_size=16, learning_rate=0.001):
    dataset = EEGDataset(processed_data_dir)
    num_classes = len(dataset.class_names)
    input_channels = dataset.samples.shape[2]

    validation_size = int(0.2 * len(dataset))
    training_size = len(dataset) - validation_size
    training_set, validation_set = random_split(dataset, [training_size, validation_size])

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)

    model = EEGClassifier1DCNN(input_channels=input_channels, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # For plotting after training
    train_losses = []
    val_accuracies = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_epoch_loss = 0

        for batch_data, batch_labels in training_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data)
            loss = loss_function(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        # Evaluate model performance on validation data
        model.eval()
        total_correct = 0
        total_samples = 0
        val_loss = 0
        predictions_list = []
        targets_list = []

        with torch.no_grad():
            for validation_data, validation_labels in validation_loader:
                validation_data, validation_labels = validation_data.to(device), validation_labels.to(device)

                validation_logits = model(validation_data)
                loss = loss_function(validation_logits, validation_labels)
                val_loss += loss.item()

                _, predicted_labels = torch.max(validation_logits, 1)
                total_samples += validation_labels.size(0)
                total_correct += (predicted_labels == validation_labels).sum().item()

                predictions_list.extend(predicted_labels.cpu().numpy())
                targets_list.extend(validation_labels.cpu().numpy())

        
        validation_accuracy = 100 * total_correct / total_samples
        train_losses.append(total_epoch_loss)
        val_losses.append(val_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_epoch_loss:.4f}, Val Acc: {validation_accuracy:.2f}%")

    # Plot accuracy and loss after training
    epochs = list(range(1, num_epochs + 1))
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:green')
    ax2.plot(epochs, val_accuracies, label='Accuracy', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title("Training Loss and Validation Accuracy Over Epochs")
    plt.tight_layout()
    plt.show()

    # Display confusion matrix after final epoch
    confusion_mat = confusion_matrix(targets_list, predictions_list)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=dataset.class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Validation Confusion Matrix (Final Epoch)")
    plt.tight_layout()
    plt.show()

    model_output_path = processed_data_dir / 'eeg_1dcnn_model.pth'
    torch.save(model.state_dict(), model_output_path)
    print(f"Model saved to {model_output_path}")

# Main entry point: prepares paths and starts training
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python torch_baseline_model_gen.py <recording_folder_name>")
        sys.exit(1)

    recording_folder = sys.argv[1]
    base_directory = Path(__file__).resolve().parent.parent.parent

    input_data_path = base_directory / 'data' / 'processed' / 'training' / recording_folder
    output_model_path = base_directory / 'data' / 'model' / recording_folder

    if not input_data_path.exists():
        print(f"Error: folder '{input_data_path}' does not exist.")
        sys.exit(1)

    os.makedirs(output_model_path, exist_ok=True)

    train_eeg_cnn_model(input_data_path)

    trained_model_file = input_data_path / 'eeg_1dcnn_model.pth'
    final_destination = output_model_path / 'eeg_1dcnn_model.pth'
    os.replace(trained_model_file, final_destination)

    print(f"Model moved to {final_destination}")
