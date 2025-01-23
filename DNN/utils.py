import os
import csv
import PIL
from PIL import *
from concurrent.futures import ThreadPoolExecutor
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from skimage.color import rgb2hsv
import numpy as np
import numpy as np
from skimage import data
from skimage.color import rgb2hed, hed2rgb, rgb2gray
import skimage


def rgb_to_hed(img):
    ihc_hed = rgb2hed(img)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    h, e, d = (
        rgb2gray(ihc_h),
        rgb2gray(ihc_e),
        rgb2gray(ihc_d),
    )  # so that all lies in [0,1]
    return h, e, d


def save_metrics_to_csv(filename, epoch, loss, val_loss, accuracy, val_accuracy):
    # Check if file exists; if not, write the header
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

        # Write the metrics for the current epoch
        writer.writerow([epoch, loss, accuracy, val_loss, val_accuracy])


# Helper function to load a single image and extract label
def load_image_and_label(img_path):
    label = int(os.path.basename(img_path).split("_")[-1].split(".")[0])
    image = Image.open(img_path).convert("RGB")
    return image, label


class PCAM_dataset(Dataset):
    def __init__(self, image_folder, transform=None, nworkers=1):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.images = []

        # Collect image paths
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)

        # Load all images into memory in parallel
        self._load_images_in_parallel(nworkers)


class PCAM_dataset_rgb(PCAM_dataset):
    def __init__(self, image_folder, transform=None, nworkers=1):
        super().__init__(image_folder, transform, nworkers)

    def _load_images_in_parallel(self, num_workers):
        # Use ThreadPoolExecutor to load images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(load_image_and_label, self.image_paths)
            for image, label in results:
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torchvision.transforms.functional.pil_to_tensor(image)
        image = image.type(torch.float32)
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, label


class PCAM_dataset_rgb_hsv(PCAM_dataset):
    def __init__(self, image_folder, transform=None, nworkers=1):
        super().__init__(image_folder, transform, nworkers)

    def _load_images_in_parallel(self, num_workers):
        # Use ThreadPoolExecutor to load images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(load_image_and_label, self.image_paths)
            for image, label in results:
                hsv_img = rgb2hsv(np.array(image))
                hsv_img = hsv_img.transpose(2, 0, 1)  # C first
                self.images.append((image, hsv_img))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_rgb, image_hsv = self.images[idx]
        label = self.labels[idx]
        image_rgb = torchvision.transforms.functional.pil_to_tensor(
            image_rgb
        )  # C first
        image_rgb = image_rgb / 255.0
        image_hsv = torch.from_numpy(
            image_hsv
        )  # C first HSV in already in 0 255 in PIL
        image = torch.stack((image_rgb, image_hsv), dim=0).view(6, 96, 96)
        image = image.type(torch.float32)
        if self.transform:
            image = self.transform(image)

        return image, label


class PCAM_dataset_rgb_hsv_hed(PCAM_dataset):
    def __init__(self, image_folder, transform=None, nworkers=1):
        super().__init__(image_folder, transform, nworkers)

    def _load_images_in_parallel(self, num_workers):
        # Use ThreadPoolExecutor to load images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(load_image_and_label, self.image_paths)
            for image, label in results:
                hsv_img = rgb2hsv(np.array(image))
                hsv_img = hsv_img.transpose(2, 0, 1)  # C first
                h, e, d = rgb_to_hed(np.array(image))
                self.images.append((image, hsv_img, h, e, d))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_rgb, image_hsv, h, e, d = self.images[idx]
        label = self.labels[idx]
        image_rgb = torchvision.transforms.functional.pil_to_tensor(
            image_rgb
        )  # C first
        image_rgb = image_rgb / 255.0
        image_hsv = torch.from_numpy(image_hsv)  # C first HSV in 0,1
        image_h = torch.from_numpy(h)
        image_e = torch.from_numpy(e)
        image_d = torch.from_numpy(d)
        hed = torch.stack((image_h, image_e, image_d))  # 3 x 96 x 96
        image = torch.stack((image_rgb, image_hsv, hed), dim=0).view(9, 96, 96)
        image = image.type(torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, label


# rgb + hsv + gray ?


class PCAM_dataset_gray(PCAM_dataset):
    def __init__(self, image_folder, transform=None, nworkers=1):
        super().__init__(image_folder, transform, nworkers)

    def _load_images_in_parallel(self, num_workers):
        # Use ThreadPoolExecutor to load images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(load_image_and_label, self.image_paths)
            for image, label in results:
                self.images.append(image.convert("L"))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_gray = self.images[idx]
        label = self.labels[idx]
        image = torchvision.transforms.functional.pil_to_tensor(image_gray)  # C first
        image = image.type(torch.float32)
        image = image / 255.0
        if self.transform:
            image = self.transform(image)

        return image, label


def calculate_statistics(dset, batch_size=1, num_workers=1):
    dloader = DataLoader(
        dset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    x, _ = dloader.dataset[0]
    num_channels = x.size(0)
    mean, std = torch.zeros(num_channels), torch.zeros(num_channels)
    last_i = 0
    for i, (x, _) in enumerate(dloader, start=0):
        x = x.permute(1, 0, 2, 3).contiguous().view(x.size(1), -1)  # (3, ..)
        mean.add_(x.mean(dim=1))
        std.add_(x.std(dim=1))
        last_i = i
    mean.div_(last_i + 1)
    std.div_(last_i + 1)
    return mean.tolist(), std.tolist()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    val_loss,
    accuracy,
    val_accuracy,
    base_filename="DLmodels/checkpoint",
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "validation_loss": val_loss,
        "accuracy": accuracy,
        "validation_accuracy": val_accuracy,
    }
    filename = base_filename + "_" + str(epoch) + ".pth"
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


# Define the CNN architecture with Dropout
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(32 * 24 * 24, 128)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        self.fc2 = nn.Linear(
            128, 2
        )  # Output layer with 2 units for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 24 * 24)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first fully connected layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # Apply log_softmax to the output


def evaluate_model(model, test_loader, criterion, device="cuda"):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = (
                images.to(device),
                labels.to(device),
            )  # Move data to the device (GPU/CPU)

            # Forward pass: compute predictions
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get predicted labels by choosing the class with the highest score
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate average loss and accuracy over the test set
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.5f}%")

    return avg_loss, accuracy
