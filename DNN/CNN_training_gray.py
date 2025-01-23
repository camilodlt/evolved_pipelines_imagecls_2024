from torchvision import *
import os
from torch.utils.data import DataLoader, Dataset
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
from utils import (
    PCAM_dataset_gray,
    calculate_statistics,
    save_checkpoint,
    save_metrics_to_csv,
    SimpleCNN,
    evaluate_model,
)
import pandas as pd

seed = 123
random.seed(seed)
torch.manual_seed(seed)

dloader_args = {"num_workers": 1, "pin_memory": False}

# Record the start time
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NWORKERS = os.cpu_count()

METRICS_FILE = "cnn_metrics_gray.csv"  # CHANGE
EPOCHS = 100  # CHANGE
CHECKPOINT_BASENAME = "DLmodels/checkpoint_gray"  # CHANGE

print(f"TORCH DEVICE {device}")
print(f"N CPUS {NWORKERS}")
print(f"CWD : {os.getcwd()}")
print(f"Metrics file: {METRICS_FILE}")

# Paths to your datasets
train_folder = "datasets/pcam/train"
val_folder = "datasets/pcam/val"
test_folder = "datasets/pcam/test"

print("WILL READ VAL DATA")
VAL_DATASET = PCAM_dataset_gray(val_folder, None, NWORKERS)  # CHANGE

print("WILL READ TRAIN DATA")
TRAIN_DATASET = PCAM_dataset_gray(train_folder, None, NWORKERS)  # CHANGE

print(f"TRAIN DATASET SIZE : {len(TRAIN_DATASET)}")
print(f"VAL DATASET SIZE : {len(VAL_DATASET)}")

train_mean, train_std = calculate_statistics(
    TRAIN_DATASET, batch_size=128, num_workers=1
)

print(f"normalization mean: {train_mean} | std: {train_std}")

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(train_mean, train_std),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Normalize(train_mean, train_std),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Normalize(train_mean, train_std),
    ]
)

TRAIN_DATASET.transform = train_transform
VAL_DATASET.transform = val_transform

train_dataloader = DataLoader(
    TRAIN_DATASET, batch_size=64, shuffle=True, **dloader_args
)
val_dataloader = DataLoader(VAL_DATASET, batch_size=64, shuffle=False, **dloader_args)


# Define the training loop with accuracy calculation
def train_model(
    model, train_dataloder, val_dataloder, criterion, optimizer, num_epochs=5
):
    with torch.no_grad():
        model.eval()
        epoch = 0
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_dataloder):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(
                outputs, 1
            )  # Get the index of the max log-probability
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        epoch_loss = running_loss / len(train_dataloder)
        epoch_acc = correct_train / total_train
        print(
            f"Epoch [{epoch}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}"
        )

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        for inputs, labels in val_dataloder:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
        val_loss /= len(val_dataloder)
        val_acc = correct_val / total_val
        save_metrics_to_csv(
            METRICS_FILE, epoch, epoch_loss, val_loss, epoch_acc, val_acc
        )
        print(
            f"Epoch [{epoch}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
        )

    for epoch in range(num_epochs):
        epoch += 1
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_dataloder):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the parameters

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(
                outputs, 1
            )  # Get the index of the max log-probability
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_dataloder)
        epoch_acc = correct_train / total_train

        print(
            f"Epoch [{epoch}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}"
        )

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_dataloder:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_dataloder)
        val_acc = correct_val / total_val

        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_loss,
            val_loss,
            epoch_acc,
            val_acc,
            base_filename=CHECKPOINT_BASENAME,
        )
        save_metrics_to_csv(
            METRICS_FILE, epoch, epoch_loss, val_loss, epoch_acc, val_acc
        )
        print(
            f"Epoch [{epoch}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
        )

    print("Training complete.")


# Example usage
model = SimpleCNN(1).to(device)  # CHANGE
criterion = nn.NLLLoss()  # Negative Log-Likelihood Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Assuming train_dloader and val_dloader are defined and populated with data
print("Model TRAIN begin ...")
train_model(
    model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=EPOCHS
)
print("Model TRAIN done")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

#####
# GET THE CHECKPOINT #
#####
df = pd.read_csv(METRICS_FILE)
max_val_accuracy_idx = df["val_accuracy"].idxmax()
best_checkpoint = CHECKPOINT_BASENAME + "_" + str(max_val_accuracy_idx) + ".pth"
print(df)
print("MAX acc : ", max_val_accuracy_idx)
print("Checkpoint to read ", best_checkpoint)
state = torch.load(best_checkpoint)
model.load_state_dict(state["model_state_dict"])
print("Weights were copied")

# EVAL ON TEST
print("WILL READ TEST DATA")
TEST_DATASET = PCAM_dataset_gray(test_folder, test_transform, NWORKERS)  # CHANGE
test_loader = DataLoader(TEST_DATASET, batch_size=64, shuffle=False, **dloader_args)

evaluate_model(model, test_loader, criterion, device)
