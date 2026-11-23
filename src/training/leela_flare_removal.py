# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')





get_ipython().getoutput("pip install -q gdown")

import os
import gdown

# Create dataset archive directory
os.makedirs("/kaggle/working/data/archives", exist_ok=True)

# Dataset paths
flare_zip = "/kaggle/working/data/archives/Flare7K++.zip"
flickr_zip = "/kaggle/working/data/archives/Flickr24K.zip"

# Download Flare7K++
gdown.download(
    id="1tJbK97sNLakOll6CJYU66AO_9E0xvA_0",
    output=flare_zip,
    quiet=False
)

# Download Flickr24K
gdown.download(
    id="1mRPLMijWWtQ9-_6-f7wYgSyskPM6vb4G",
    output=flickr_zip,
    quiet=False
)

# Verify downloaded files
print("\n========== DOWNLOAD VERIFICATION ==========")

for path in [flare_zip, flickr_zip]:
    if os.path.exists(path):
        print(os.path.basename(path))
        print("Size:", round(os.path.getsize(path) / (1024**3), 3), "GB")
    else:
        print("DOWNLOAD FAILED:", path)


import os
import zipfile
import shutil

archive_dir = "/kaggle/working/data/archives"

flare_zip = os.path.join(archive_dir, "Flare7K++.zip")
flickr_zip = os.path.join(archive_dir, "Flickr24K.zip")

print("========== STORAGE CHECK ==========")

total, used, free = shutil.disk_usage("/kaggle/working")

print("Total storage:", round(total / (1024**3), 2), "GB")
print("Used storage:", round(used / (1024**3), 2), "GB")
print("Free storage:", round(free / (1024**3), 2), "GB")

print("\n========== ZIP INSPECTION ==========")

for path in [flare_zip, flickr_zip]:

    print("\nFile:", os.path.basename(path))
    print("Exists:", os.path.exists(path))

    with zipfile.ZipFile(path, "r") as z:
        files = z.namelist()

        print("ZIP integrity:", "Checking...")
        print("Total entries:", len(files))
        print("First 10 entries:")

        for name in files[:10]:
            print(" ", name)

        print("ZIP integrity result:", z.testzip())


import zipfile
import os

archive_dir = "/kaggle/working/data/archives"

for filename in ["Flare7K++.zip", "Flickr24K.zip"]:

    path = os.path.join(archive_dir, filename)

    with zipfile.ZipFile(path, "r") as z:
        total_size = sum(info.file_size for info in z.infolist())

    print(filename)
    print("Compressed size:", round(os.path.getsize(path) / (1024**3), 2), "GB")
    print("Extracted size:", round(total_size / (1024**3), 2), "GB")
    print()


import os
import zipfile

data_dir = "/kaggle/working/data"
archive_dir = os.path.join(data_dir, "archives")

# Extract Flare7K++
print("Extracting Flare7K++...")
with zipfile.ZipFile(os.path.join(archive_dir, "Flare7K++.zip"), "r") as z:
    z.extractall(data_dir)

print("Flare7K++ extraction completed!")

# Extract Flickr24K
print("\nExtracting Flickr24K...")
with zipfile.ZipFile(os.path.join(archive_dir, "Flickr24K.zip"), "r") as z:
    z.extractall(data_dir)

print("Flickr24K extraction completed!")

print("\nBoth datasets extracted!")


import os

data_dir = "/kaggle/working/data"

print("========== DATASET FOLDERS ==========")

for folder in ["Flare7Kpp", "Flickr24K"]:
    path = os.path.join(data_dir, folder)

    print(f"\n{folder}")
    print("Exists:", os.path.exists(path))

    if os.path.exists(path):
        print("Contents:")
        for item in os.listdir(path)[:15]:
            print(" -", item)

print("\n========== IMAGE COUNTS ==========")

flickr_path = os.path.join(data_dir, "Flickr24K")

flickr_count = sum(
    1 for f in os.listdir(flickr_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

print("Flickr24K images:", flickr_count)

flare_path = os.path.join(data_dir, "Flare7Kpp", "Flare7K")

for folder in [
    "Scattering_Flare/Compound_Flare",
    "Reflective_Flare"
]:
    path = os.path.join(flare_path, folder)

    count = sum(
        1 for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    print(folder, ":", count)


import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    ), "GB")
else:
    print("WARNING: GPU is not available!")


get_ipython().getoutput("wget -q https://raw.githubusercontent.com/ykdai/Flare7K/main/data_loader.py -O /kaggle/working/data_loader.py")

import os

print("Data loader downloaded:", os.path.exists("/kaggle/working/data_loader.py"))
print("File size:", os.path.getsize("/kaggle/working/data_loader.py"), "bytes")


with open("/kaggle/working/data_loader.py", "r") as f:
    code = f.read()

print(code)


import torch
import torchvision.transforms as transforms

IMAGE_SIZE = 256

transform_base_256 = transforms.Compose([
    transforms.RandomCrop(
        (IMAGE_SIZE, IMAGE_SIZE),
        pad_if_needed=True,
        padding_mode="reflect"
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

transform_flare_256 = transforms.Compose([
    transforms.RandomAffine(
        degrees=(0, 360),
        scale=(0.8, 1.5),
        translate=(300/1440, 300/1440),
        shear=(-20, 20)
    ),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

print("Image transformations configured!")
print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)


import os
from data_loader import Flare_Image_Loader

FLICKR_ROOT = "/kaggle/working/data/Flickr24K"
FLARE_ROOT = "/kaggle/working/data/Flare7Kpp/Flare7K"

flare_image_loader_256 = Flare_Image_Loader(
    FLICKR_ROOT,
    transform_base_256,
    transform_flare_256,
    mask_type="luminance"
)

flare_image_loader_256.load_scattering_flare(
    "Flare7K",
    os.path.join(
        FLARE_ROOT,
        "Scattering_Flare",
        "Compound_Flare"
    )
)

flare_image_loader_256.load_reflective_flare(
    "Flare7K",
    os.path.join(
        FLARE_ROOT,
        "Reflective_Flare"
    )
)

print("\n========== DATASET SUMMARY ==========")
print("Base images:", len(flare_image_loader_256))
print("Scattering flares:", len(flare_image_loader_256.flare_list))
print("Reflective flares:", len(flare_image_loader_256.reflective_list))


sample = flare_image_loader_256[0]

base_img, flare_img, merge_img, flare_mask, gamma = sample

print("========== SAMPLE VERIFICATION ==========")
print("Base image shape:", base_img.shape)
print("Flare image shape:", flare_img.shape)
print("Merge image shape:", merge_img.shape)
print("Flare mask shape:", flare_mask.shape)
print("Gamma value:", gamma)

print("\nData type:", base_img.dtype)
print("Base image range:", base_img.min().item(), "to", base_img.max().item())
print("Merge image range:", merge_img.min().item(), "to", merge_img.max().item())


from torch.utils.data import random_split

TRAIN_SIZE = 10000
VAL_SIZE = 2500
UNUSED_SIZE = len(flare_image_loader_256) - TRAIN_SIZE - VAL_SIZE

train_dataset, val_dataset, unused_dataset = random_split(
    flare_image_loader_256,
    [TRAIN_SIZE, VAL_SIZE, UNUSED_SIZE],
    generator=torch.Generator().manual_seed(42)
)

print("========== DATASET SPLIT ==========")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Unused images:", len(unused_dataset))
print("Total:", len(train_dataset) + len(val_dataset) + len(unused_dataset))


from torch.utils.data import DataLoader

BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Batch size:", BATCH_SIZE)


import torch

device = torch.device("cuda")

# Load one real batch
base_img, flare_img, merge_img, flare_mask, gamma = next(iter(train_loader))

# Move model input and target to GPU
inputs = merge_img.to(device)
targets = base_img.to(device)

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU allocated:", round(torch.cuda.memory_allocated() / 1024**3, 3), "GB")
print("GPU reserved:", round(torch.cuda.memory_reserved() / 1024**3, 3), "GB")

del inputs, targets, base_img, flare_img, merge_img, flare_mask, gamma

torch.cuda.empty_cache()

print("\nBatch memory test completed!")


import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.final(d1))


model = UNet().to(device)

total_params = sum(p.numel() for p in model.parameters())

print("U-Net architecture created successfully!")
print("Device:", device)
print("Parameters:", total_params)


import torch
import torch.nn as nn

criterion = nn.L1Loss()

scaler = torch.amp.GradScaler("cuda")

# Get one batch
base_img, _, merge_img, _, _ = next(iter(train_loader))

inputs = merge_img.to(device)
targets = base_img.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Mixed precision forward pass
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Backward pass
optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print("Forward and backward pass successful!")
print("Output shape:", outputs.shape)
print("Test loss:", loss.item())
print("GPU peak memory:", round(
    torch.cuda.max_memory_allocated() / (1024**3), 2
), "GB")

# Clean up test batch
del inputs, targets, outputs, loss, base_img, merge_img
torch.cuda.empty_cache()


import os

PROJECT_DIR = "/kaggle/working/Flare_Removal_Project"

CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "baseline_256")
RESULT_DIR = os.path.join(PROJECT_DIR, "results", "baseline_256")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("========== PROJECT DIRECTORIES ==========")
print("Project:", PROJECT_DIR)
print("Checkpoints:", CHECKPOINT_DIR)
print("Results:", RESULT_DIR)

print("\nCheckpoint interval: 100 batches")
print("Image size: 256 x 256")
print("Batch size: 8")
print("Planned epochs: 3")


import os
import torch
import shutil
import random
import numpy as np

# Create a fresh model
model = UNet().to(device)

# Training configuration
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")

EPOCHS = 3
CHECKPOINT_INTERVAL = 100
best_val_loss = float("inf")

# Checkpoint paths
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "baseline_unet_256_latest.pth")
BACKUP_PATH = os.path.join(CHECKPOINT_DIR, "baseline_unet_256_backup.pth")

def save_checkpoint(epoch, batch, loss):
    checkpoint = {
        "epoch": epoch,
        "batch": batch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
    }

    temp_path = LATEST_PATH + ".tmp"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, LATEST_PATH)

    shutil.copy2(LATEST_PATH, BACKUP_PATH)

print("Fresh U-Net initialized!")
print("Optimizer: Adam")
print("Learning rate: 0.0001")
print("Loss function: L1 Loss")
print("Mixed precision: Enabled")
print("Epochs:", EPOCHS)
print("Checkpoint interval:", CHECKPOINT_INTERVAL)
print("Checkpoint saving function: Ready")


def validate_model(model, val_loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for base_img, _, merge_img, _, _ in val_loader:

            inputs = merge_img.to(device, non_blocking=True)
            targets = base_img.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            batch_size = inputs.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    average_loss = total_loss / total_samples

    model.train()

    return average_loss


print("Validation function created successfully!")
print("Validation batches:", len(val_loader))
print("Validation images:", len(val_dataset))


BEST_PATH = os.path.join(
    CHECKPOINT_DIR,
    "baseline_unet_256_best.pth"
)

def save_best_model(epoch, val_loss):

    global best_val_loss

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss
        }

        torch.save(checkpoint, BEST_PATH)

        print("\nNew best model saved!")
        print("Epoch:", epoch + 1)
        print("Validation loss:", round(val_loss, 6))
        print("Path:", BEST_PATH)

    else:
        print("\nNo improvement in validation loss.")
        print("Best validation loss:", round(best_val_loss, 6))


print("Best-model saving function created!")
print("Best checkpoint path:", BEST_PATH)


def train_model():

    global best_val_loss

    training_history = []

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

        for batch_idx, (base_img, _, merge_img, _, _) in enumerate(train_loader):

            inputs = merge_img.to(device, non_blocking=True)
            targets = base_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            current_batch = batch_idx + 1

            if current_batch % 100 == 0:

                avg_loss = running_loss / 100

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{current_batch}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.6f}"
                )

                save_checkpoint(
                    epoch,
                    current_batch,
                    avg_loss
                )

                running_loss = 0.0

        # Epoch training loss
        train_loss = running_loss / (
            len(train_loader) % 100
            if len(train_loader) % 100 != 0
            else 100
        )

        print("\nRunning validation...")

        val_loss = validate_model(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.6f}")

        save_best_model(epoch, val_loss)

        # Save epoch checkpoint
        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"baseline_unet_256_epoch_{epoch + 1}.pth"
        )

        save_checkpoint(epoch, len(train_loader), train_loss)

        shutil.copy2(LATEST_PATH, epoch_path)

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f"Epoch {epoch + 1} completed!")

    print("\n========== TRAINING COMPLETED ==========")

    for item in training_history:
        print(item)

    return training_history


print("Training function defined successfully!")
print("Training has NOT started yet.")


def train_model():

    global best_val_loss

    training_history = []

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0
        total_samples = 0

        print(f"\n========== EPOCH {epoch + 1}/{EPOCHS} ==========")

        for batch_idx, (base_img, _, merge_img, _, _) in enumerate(train_loader):

            inputs = merge_img.to(device, non_blocking=True)
            targets = base_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = inputs.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            current_batch = batch_idx + 1

            if current_batch % CHECKPOINT_INTERVAL == 0:
                avg_loss = epoch_loss / total_samples

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{current_batch}/{len(train_loader)}] "
                    f"Average Loss: {avg_loss:.6f}"
                )

                save_checkpoint(epoch, current_batch, avg_loss)

        train_loss = epoch_loss / total_samples

        print("\nRunning validation...")

        val_loss = validate_model(
            model, val_loader, criterion, device
        )

        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.6f}")
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.6f}")

        save_best_model(epoch, val_loss)

        save_checkpoint(epoch, len(train_loader), train_loss)

        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"baseline_unet_256_epoch_{epoch + 1}.pth"
        )

        shutil.copy2(LATEST_PATH, epoch_path)

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f"Epoch {epoch + 1} completed!")

    print("\n========== TRAINING COMPLETED ==========")

    for item in training_history:
        print(item)

    return training_history


print("Corrected training function defined!")
print("Training has NOT started.")


def load_checkpoint(checkpoint_path):

    global best_val_loss

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting fresh training.")
        return 0, 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    saved_epoch = checkpoint["epoch"]
    saved_batch = checkpoint["batch"]

    print("Checkpoint restored successfully!")
    print("Saved epoch:", saved_epoch + 1)
    print("Saved batch:", saved_batch)
    print("Best validation loss:", best_val_loss)

    return saved_epoch, saved_batch


print("Checkpoint recovery function created!")


def save_checkpoint(
    epoch,
    batch,
    loss,
    epoch_order=None,
    epoch_loss_sum=0.0,
    total_samples=0
):
    checkpoint = {
        "epoch": epoch,
        "batch": batch,
        "loss": loss,
        "epoch_order": epoch_order,
        "epoch_loss_sum": epoch_loss_sum,
        "total_samples": total_samples,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
    }

    temp_path = LATEST_PATH + ".tmp"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, LATEST_PATH)
    shutil.copy2(LATEST_PATH, BACKUP_PATH)


def load_checkpoint(checkpoint_path):

    global best_val_loss

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting fresh.")
        return 0, 0, None, 0.0, 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    print("Checkpoint restored!")
    print("Epoch:", checkpoint["epoch"] + 1)
    print("Completed batches:", checkpoint["batch"])

    return (
        checkpoint["epoch"],
        checkpoint["batch"],
        checkpoint["epoch_order"],
        checkpoint["epoch_loss_sum"],
        checkpoint["total_samples"]
    )


print("Improved checkpoint saving and recovery functions ready!")


def load_checkpoint(checkpoint_path):

    global best_val_loss

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting fresh.")
        return 0, 0, None, 0.0, 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_val_loss = checkpoint.get(
        "best_val_loss", float("inf")
    )

    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])

    # Keep CPU RNG state on CPU
    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    # Restore CUDA RNG states
    cuda_states = [
        state.cpu() for state in checkpoint["cuda_rng_state"]
    ]
    torch.cuda.set_rng_state_all(cuda_states)

    print("Checkpoint restored successfully!")
    print("Saved epoch:", checkpoint["epoch"] + 1)
    print("Saved batch:", checkpoint["batch"])

    return (
        checkpoint["epoch"],
        checkpoint["batch"],
        checkpoint.get("epoch_order"),
        checkpoint.get("epoch_loss_sum", 0.0),
        checkpoint.get("total_samples", 0)
    )


print("Corrected checkpoint recovery function ready!")


start_epoch, start_batch, saved_order, saved_loss, saved_samples = load_checkpoint(
    LATEST_PATH
)

print("\n========== RECOVERY TEST ==========")
print("Start epoch:", start_epoch)
print("Start batch:", start_batch)
print("Recovery test completed!")



import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. TRAINING CONFIGURATION
# ==========================================

EPOCHS = 3
BATCH_SIZE = 8
CHECKPOINT_INTERVAL = 100
LEARNING_RATE = 1e-4

device = torch.device("cuda")

LATEST_PATH = os.path.join(
    CHECKPOINT_DIR, "baseline_unet_256_latest.pth"
)

BACKUP_PATH = os.path.join(
    CHECKPOINT_DIR, "baseline_unet_256_backup.pth"
)

BEST_PATH = os.path.join(
    CHECKPOINT_DIR, "baseline_unet_256_best.pth"
)

# ==========================================
# 2. CHECKPOINT SAVING
# ==========================================

def save_checkpoint(
    epoch,
    batch,
    epoch_order,
    epoch_loss_sum,
    total_samples,
    history
):

    checkpoint = {
        "epoch": epoch,
        "batch": batch,
        "epoch_order": epoch_order,
        "epoch_loss_sum": epoch_loss_sum,
        "total_samples": total_samples,
        "history": history,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        "best_val_loss": best_val_loss,

        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
    }

    temp_path = LATEST_PATH + ".tmp"

    # Atomic latest checkpoint
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, LATEST_PATH)

    # Atomic backup checkpoint
    backup_temp = BACKUP_PATH + ".tmp"
    shutil.copy2(LATEST_PATH, backup_temp)
    os.replace(backup_temp, BACKUP_PATH)


# ==========================================
# 3. CHECKPOINT RESTORATION
# ==========================================

def load_checkpoint(path):

    global best_val_loss

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    best_val_loss = checkpoint["best_val_loss"]

    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    torch.cuda.set_rng_state_all([
        state.cpu()
        for state in checkpoint["cuda_rng_state"]
    ])

    print("Checkpoint restored successfully!")
    print("Epoch:", checkpoint["epoch"] + 1)
    print("Completed batches:", checkpoint["batch"])

    return (
        checkpoint["epoch"],
        checkpoint["batch"],
        checkpoint["epoch_order"],
        checkpoint["epoch_loss_sum"],
        checkpoint["total_samples"],
        checkpoint["history"]
    )


# ==========================================
# 4. SAVE BEST MODEL
# ==========================================

def save_best_model(epoch, val_loss):

    global best_val_loss

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict()
        }

        temp_path = BEST_PATH + ".tmp"

        torch.save(best_checkpoint, temp_path)
        os.replace(temp_path, BEST_PATH)

        print("New best model saved!")
        print("Validation loss:", round(val_loss, 6))

    else:
        print("No improvement in validation loss.")


# ==========================================
# 5. VALIDATION FUNCTION
# ==========================================

def validate_model():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for base_img, _, merge_img, _, _ in val_loader:

            inputs = merge_img.to(device)
            targets = base_img.to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(inputs)
                loss = criterion(outputs, targets)

            batch_size = inputs.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()

    return total_loss / total_samples


# ==========================================
# 6. MAIN TRAINING FUNCTION
# ==========================================

def train_model(resume=False):

    global model, optimizer, scaler, best_val_loss

    # Fresh training or recovery
    if resume:

        if not os.path.exists(LATEST_PATH):
            raise FileNotFoundError("No latest checkpoint found!")

        (
            start_epoch,
            start_batch,
            saved_order,
            epoch_loss_sum,
            total_samples,
            history
        ) = load_checkpoint(LATEST_PATH)

        resume_rng = True

    else:

        # Start with completely fresh weights
        model = UNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")

        history = []

        start_epoch = 0
        start_batch = 0
        saved_order = None
        epoch_loss_sum = 0.0
        total_samples = 0

        resume_rng = False

        # Remove previous test checkpoints
        for path in [LATEST_PATH, BACKUP_PATH, BEST_PATH]:
            if os.path.exists(path):
                os.remove(path)

    # Validation loader with single-process loading
    global val_loader

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    number_of_batches = len(train_dataset) // BATCH_SIZE

    print("\n========== TRAINING CONFIGURATION ==========")
    print("Device:", torch.cuda.get_device_name(0))
    print("Epochs:", EPOCHS)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Training batches per epoch:", number_of_batches)
    print("Checkpoint interval:", CHECKPOINT_INTERVAL)

    # ======================================
    # EPOCH LOOP
    # ======================================

    for epoch in range(start_epoch, EPOCHS):

        model.train()

        # Recover exact sample order
        if epoch == start_epoch and saved_order is not None:

            epoch_order = saved_order

        else:

            generator = torch.Generator()
            generator.manual_seed(2026 + epoch)

            epoch_order = torch.randperm(
                len(train_dataset),
                generator=generator
            ).tolist()

            start_batch = 0
            epoch_loss_sum = 0.0
            total_samples = 0

        # Continue from next unfinished batch
        remaining_indices = epoch_order[
            start_batch * BATCH_SIZE:
        ]

        epoch_subset = Subset(
            train_dataset,
            remaining_indices
        )

        epoch_loader = DataLoader(
            epoch_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

        iterator = iter(epoch_loader)

        # Restore RNG after creating iterator.
        # This preserves the next augmentation sequence.
        if resume_rng:

            random.setstate(resume_rng_state[0])
            np.random.set_state(resume_rng_state[1])
            torch.set_rng_state(resume_rng_state[2])
            torch.cuda.set_rng_state_all(resume_rng_state[3])

            resume_rng = False

        elif start_batch == 0:

            save_checkpoint(
                epoch,
                0,
                epoch_order,
                epoch_loss_sum,
                total_samples,
                history
            )

        print(
            f"\n========== EPOCH {epoch + 1}/{EPOCHS} =========="
        )

        # ==================================
        # BATCH LOOP
        # ==================================

        for batch_idx, batch_data in enumerate(iterator):

            base_img, _, merge_img, _, _ = batch_data

            inputs = merge_img.to(device, non_blocking=True)

            targets = base_img.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = inputs.size(0)

            epoch_loss_sum += loss.item() * batch_size
            total_samples += batch_size

            current_batch = start_batch + batch_idx + 1

            # Periodic checkpoint
            if (
                current_batch % CHECKPOINT_INTERVAL == 0
                or current_batch == number_of_batches
            ):

                average_loss = epoch_loss_sum / total_samples

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{current_batch}/{number_of_batches}] "
                    f"Loss: {average_loss:.6f}"
                )

                save_checkpoint(
                    epoch,
                    current_batch,
                    epoch_order,
                    epoch_loss_sum,
                    total_samples,
                    history
                )

        # ==================================
        # EPOCH VALIDATION
        # ==================================

        train_loss = epoch_loss_sum / total_samples

        print("\nRunning validation...")

        val_loss = validate_model()

        print("\n========== EPOCH RESULTS ==========")
        print("Epoch:", epoch + 1)
        print("Training loss:", round(train_loss, 6))
        print("Validation loss:", round(val_loss, 6))

        save_best_model(epoch, val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # Save updated history and progress.
        # A restart here may repeat validation, but will not
        # repeat the completed training batches.
        save_checkpoint(
            epoch,
            number_of_batches,
            epoch_order,
            epoch_loss_sum,
            total_samples,
            history
        )

        print("Epoch checkpoint secured!")

        # Prepare next epoch
        start_batch = 0
        saved_order = None
        epoch_loss_sum = 0.0
        total_samples = 0

    print("\n========== TRAINING FINISHED ==========")

    for result in history:
        print(result)

    print("\nLatest checkpoint:", LATEST_PATH)
    print("Backup checkpoint:", BACKUP_PATH)
    print("Best model:", BEST_PATH)

    return history


print("Reliable training system configured!")
print("Training has NOT started yet.")



training_history = train_model(resume=False)


import os
import shutil

CHECKPOINT_DIR = "/kaggle/working/Flare_Removal_Project/checkpoints/baseline_256"

print("Checkpoint files:")
for filename in os.listdir(CHECKPOINT_DIR):
    path = os.path.join(CHECKPOINT_DIR, filename)
    print(filename, "-", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")

ZIP_PATH = shutil.make_archive(
    "/kaggle/working/Flare_Removal_Checkpoints_3Epochs",
    "zip",
    CHECKPOINT_DIR
)

print("\nBackup ZIP created:", ZIP_PATH)
print("ZIP size:", round(os.path.getsize(ZIP_PATH) / (1024 * 1024), 2), "MB")


from IPython.display import FileLink, display

ZIP_PATH = "/kaggle/working/Flare_Removal_Checkpoints_3Epochs.zip"

display(FileLink(ZIP_PATH))


import os

TEST_ROOT = "/kaggle/working/data/Flare7Kpp/test_data"

print("Test directory exists:", os.path.exists(TEST_ROOT))

for root, dirs, files in os.walk(TEST_ROOT):
    level = root.replace(TEST_ROOT, "").count(os.sep)

    if level <= 3:
        print("  " * level + os.path.basename(root) + "/")

        for file in files[:5]:
            print("  " * (level + 1) + file)


import os
from PIL import Image

TEST_ROOT = "/kaggle/working/data/Flare7Kpp/test_data"

for category in ["synthetic", "real"]:
    print(f"\n========== {category.upper()} DATA ==========")

    for folder in ["input", "gt", "mask"]:
        folder_path = os.path.join(TEST_ROOT, category, folder)

        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"{folder}: {len(files)} images")

        if files:
            image_path = os.path.join(folder_path, files[0])

            with Image.open(image_path) as img:
                print("  Example:", files[0])
                print("  Dimensions:", img.size)
                print("  Mode:", img.mode)

    input_files = {
        f.replace("input_", "")
        for f in os.listdir(os.path.join(TEST_ROOT, category, "input"))
        if f.startswith("input_")
    }

    gt_files = {
        f.replace("gt_", "")
        for f in os.listdir(os.path.join(TEST_ROOT, category, "gt"))
        if f.startswith("gt_")
    }

    print("Matching input-GT filenames:", len(input_files & gt_files))



import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        output = self.final(d1)

        return self.sigmoid(output)


# Verify architecture
model_check = UNet()

print("U-Net architecture loaded successfully!")
print("Parameters:", sum(p.numel() for p in model_check.parameters()))



import os
import torch

BEST_PATH = "/kaggle/working/Flare_Removal_Project/checkpoints/baseline_256/baseline_unet_256_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = UNet().to(device)

# Load best checkpoint
checkpoint = torch.load(
    BEST_PATH,
    map_location=device,
    weights_only=False
)

print("Checkpoint keys:", checkpoint.keys())

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nBest trained model loaded successfully!")
print("Device:", device)
print("Validation loss: 0.050341")



import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

TEST_ROOT = "/kaggle/working/data/Flare7Kpp/test_data"

# Select one synthetic test image
category = "synthetic"
image_number = 0

filename = f"{image_number:06d}.png"

input_path = os.path.join(
    TEST_ROOT, category, "input", f"input_{filename}"
)

gt_path = os.path.join(
    TEST_ROOT, category, "gt", f"gt_{filename}"
)

# Load input and ground truth
input_img = Image.open(input_path).convert("RGB")
gt_img = Image.open(gt_path).convert("RGB")

# Resize input to the model's training resolution
input_resized = input_img.resize((256, 256))

input_tensor = transforms.ToTensor()(input_resized)
input_tensor = input_tensor.unsqueeze(0).to(device)

# Generate prediction
with torch.no_grad():
    prediction = model(input_tensor)

# Resize prediction to original resolution
prediction = F.interpolate(
    prediction,
    size=(512, 512),
    mode="bilinear",
    align_corners=False
)

prediction = prediction.squeeze(0).cpu().clamp(0, 1)

# Convert to NumPy
input_array = np.array(input_img)
gt_array = np.array(gt_img)
output_array = prediction.permute(1, 2, 0).numpy()

# Display results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(input_array)
axes[0].set_title("Input - Flare Image")

axes[1].imshow(gt_array)
axes[1].set_title("Ground Truth - Clean Image")

axes[2].imshow(output_array)
axes[2].set_title("U-Net Model Output")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()




import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# All images must be in the [0, 1] range
input_float = input_array.astype(np.float32) / 255.0
gt_float = gt_array.astype(np.float32) / 255.0
output_float = output_array.astype(np.float32)

# Calculate errors
input_mae = np.mean(np.abs(input_float - gt_float))
output_mae = np.mean(np.abs(output_float - gt_float))

input_psnr = peak_signal_noise_ratio(gt_float, input_float, data_range=1.0)
output_psnr = peak_signal_noise_ratio(gt_float, output_float, data_range=1.0)

print("========== IMAGE QUALITY EVALUATION ==========")
print(f"Input MAE:  {input_mae:.6f}")
print(f"Output MAE: {output_mae:.6f}")
print()
print(f"Input PSNR:  {input_psnr:.4f} dB")
print(f"Output PSNR: {output_psnr:.4f} dB")
print()
print("Input mean brightness:", input_float.mean())
print("Ground truth mean brightness:", gt_float.mean())
print("Output mean brightness:", output_float.mean())

if output_psnr > input_psnr:
    print("\nModel improved the PSNR.")
else:
    print("\nModel did NOT improve the PSNR.")




import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

# Load original 512x512 image directly
input_512 = transforms.ToTensor()(input_img).unsqueeze(0).to(device)

# Predict at original resolution
model.eval()

with torch.no_grad():
    prediction_512 = model(input_512)

prediction_512 = prediction_512.squeeze(0).cpu().clamp(0, 1)

# Convert to NumPy
direct_output = prediction_512.permute(1, 2, 0).numpy()

input_float = np.array(input_img).astype(np.float32) / 255.0
gt_float = np.array(gt_img).astype(np.float32) / 255.0

# Calculate MAE manually
input_mae = np.mean(np.abs(input_float - gt_float))
direct_mae = np.mean(np.abs(direct_output - gt_float))

# Calculate PSNR
input_psnr = peak_signal_noise_ratio(
    gt_float, input_float, data_range=1.0
)

direct_psnr = peak_signal_noise_ratio(
    gt_float, direct_output, data_range=1.0
)

print("========== DIRECT 512x512 EVALUATION ==========")
print(f"Input MAE:  {input_mae:.6f}")
print(f"Output MAE: {direct_mae:.6f}")
print(f"Input PSNR:  {input_psnr:.4f} dB")
print(f"Output PSNR: {direct_psnr:.4f} dB")

# Display comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(input_float)
axes[0].set_title("Original Input - 512x512")

axes[1].imshow(gt_float)
axes[1].set_title("Ground Truth")

axes[2].imshow(direct_output)
axes[2].set_title("Direct Model Output - 512x512")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()




import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio

TEST_ROOT = "/kaggle/working/data/Flare7Kpp/test_data/synthetic"

to_tensor = transforms.ToTensor()

input_maes = []
output_maes = []
input_psnrs = []
output_psnrs = []

model.eval()

for i in range(100):
    name = f"{i:06d}.png"

    input_img_eval = Image.open(
        os.path.join(TEST_ROOT, "input", f"input_{name}")
    ).convert("RGB")

    gt_img_eval = Image.open(
        os.path.join(TEST_ROOT, "gt", f"gt_{name}")
    ).convert("RGB")

    # Evaluate at the model's training resolution
    input_256 = input_img_eval.resize((256, 256))
    gt_256 = gt_img_eval.resize((256, 256))

    x = to_tensor(input_256).unsqueeze(0).to(device)
    gt = np.array(gt_256).astype(np.float32) / 255.0

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze(0).cpu().clamp(0, 1)
    pred = pred.permute(1, 2, 0).numpy()

    original = np.array(input_256).astype(np.float32) / 255.0

    input_maes.append(np.mean(np.abs(original - gt)))
    output_maes.append(np.mean(np.abs(pred - gt)))

    input_psnrs.append(
        peak_signal_noise_ratio(gt, original, data_range=1.0)
    )
    output_psnrs.append(
        peak_signal_noise_ratio(gt, pred, data_range=1.0)
    )

print("========== 100 SYNTHETIC TEST IMAGES ==========")
print("Average Input MAE:", np.mean(input_maes))
print("Average Model MAE:", np.mean(output_maes))
print("Average Input PSNR:", np.mean(input_psnrs))
print("Average Model PSNR:", np.mean(output_psnrs))

print("\nImages improved by model:",
      sum(o < i for o, i in zip(output_maes, input_maes)), "/ 100")



# Inspect the actual attributes used by your loader

print("========== LOADER ATTRIBUTES ==========")

for name, value in vars(flare_image_loader_256).items():
    try:
        print(name, ":", type(value).__name__, "| Length:", len(value))
    except TypeError:
        print(name, ":", type(value).__name__)

print("\n========== ONE TRAINING SAMPLE ==========")

sample = flare_image_loader_256[0]

print("Sample type:", type(sample))

if isinstance(sample, (tuple, list)):
    print("Number of returned items:", len(sample))

    for i, item in enumerate(sample):
        if torch.is_tensor(item):
            print(f"Item {i}: Shape={item.shape}, "
                  f"Min={item.min().item():.4f}, "
                  f"Max={item.max().item():.4f}")
        else:
            print(f"Item {i}: {type(item).__name__}, {item}")



import matplotlib.pyplot as plt
import torch

fig, axes = plt.subplots(5, 3, figsize=(12, 18))

for i in range(5):
    sample = flare_image_loader_256[i]

    base, flare, merged, mask, gamma = sample

    print(f"\nSample {i}")
    print("Gamma:", gamma)

    print("Base range:",
          base.min().item(), base.max().item())

    print("Flare range:",
          flare.min().item(), flare.max().item())

    print("Merged range:",
          merged.min().item(), merged.max().item())

    print("Mask range:",
          mask.min().item(), mask.max().item())

    images = [base, merged, mask]
    titles = ["Clean Base", "Merged Input", "Flare Mask"]

    for j in range(3):
        axes[i, j].imshow(
            images[j].permute(1, 2, 0).clamp(0, 1).numpy()
        )
        axes[i, j].set_title(f"Sample {i}: {titles[j]}")
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()




import torch
import numpy as np

print("========== TRAINING DATA DIAGNOSTICS ==========")

for i in range(20):

    base, flare, merged, mask, gamma = flare_image_loader_256[i]

    # Difference between clean and merged images
    difference = torch.abs(merged - base)

    # Calculate statistics
    flare_difference = difference.mean().item()
    mask_coverage = (mask > 0.1).float().mean().item()
    flare_max = flare.max().item()
    merged_max = merged.max().item()

    print(
        f"Sample {i:02d} | "
        f"Difference: {flare_difference:.5f} | "
        f"Mask coverage: {mask_coverage*100:.2f}% | "
        f"Flare max: {flare_max:.3f} | "
        f"Merged max: {merged_max:.3f}"
    )



import inspect

print("========== FLARE IMAGE LOADER ==========")

print(inspect.getsource(Flare_Image_Loader.__getitem__))

print("\n========== CLASS INITIALIZATION ==========")

print(inspect.getsource(Flare_Image_Loader.__init__))


import inspect

print("========== TRAINING FUNCTIONS ==========")

# Find training-related functions in the current notebook
for name, obj in list(globals().items()):
    if callable(obj) and (
        "train" in name.lower()
        or "epoch" in name.lower()
    ):
        try:
            print("\nFUNCTION:", name)
            print(inspect.getsource(obj))
        except (OSError, TypeError):
            print("Source unavailable:", name)


import matplotlib.pyplot as plt

base, flare, merged, mask, gamma = flare_image_loader_256[0]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

images = [base, flare, merged, mask]
titles = ["Clean Image", "Flare", "Merged Input", "Flare Mask"]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

print("Gamma:", gamma)


print("Mask type:", flare_image_loader_256.mask_type)

print("\nIMAGE STATISTICS")
print("Clean image max:", base.max().item())
print("Flare max:", flare.max().item())
print("Merged image max:", merged.max().item())

print("\nMASK STATISTICS")
print("Mask minimum:", mask.min().item())
print("Mask maximum:", mask.max().item())
print("Mask mean:", mask.float().mean().item())

print(
    "Mask coverage:",
    (mask > 0).float().mean().item() * 100,
    "%"
)


import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import os

TEST_ROOT = "/kaggle/working/data/Flare7Kpp/test_data/synthetic"

fig, axes = plt.subplots(3, 3, figsize=(14, 12))

model.eval()

for row, i in enumerate([0, 1, 2]):
    name = f"{i:06d}.png"

    inp = Image.open(
        os.path.join(TEST_ROOT, "input", f"input_{name}")
    ).convert("RGB")

    gt = Image.open(
        os.path.join(TEST_ROOT, "gt", f"gt_{name}")
    ).convert("RGB")

    inp256 = inp.resize((256, 256))
    gt256 = gt.resize((256, 256))

    x = transforms.ToTensor()(inp256).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze().cpu().clamp(0, 1)
    pred = pred.permute(1, 2, 0).numpy()

    axes[row, 0].imshow(inp256)
    axes[row, 0].set_title("Original Input")

    axes[row, 1].imshow(pred)
    axes[row, 1].set_title("Model Output")

    axes[row, 2].imshow(gt256)
    axes[row, 2].set_title("Ground Truth")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.show()


import torch
from torch.utils.data import random_split, DataLoader

# Recreate the same dataset split
TRAIN_SIZE = 10000
VAL_SIZE = 2500
UNUSED_SIZE = len(flare_image_loader_256) - TRAIN_SIZE - VAL_SIZE

train_dataset, val_dataset, unused_dataset = random_split(
    flare_image_loader_256,
    [TRAIN_SIZE, VAL_SIZE, UNUSED_SIZE],
    generator=torch.Generator().manual_seed(42)
)

# Recreate DataLoaders
BATCH_SIZE = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False
)

print("Training loader restored:", len(train_loader), "batches")
print("Validation loader restored:", len(val_loader), "batches")


import torch
import torch.nn as nn

model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5
)

criterion = nn.L1Loss()
scaler = torch.amp.GradScaler("cuda")

total_loss = 0
max_batches = 500

for batch_idx, (base, flare, merged, mask, gamma) in enumerate(train_loader):

    if batch_idx >= max_batches:
        break

    inputs = merged.to(device)
    targets = base.to(device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss.item()

    if (batch_idx + 1) % 100 == 0:
        print(
            f"Batch {batch_idx + 1}/{max_batches} | "
            f"Loss: {loss.item():.6f}"
        )

model.eval()

torch.save(
    model.state_dict(),
    "/kaggle/working/flare_finetuned_500.pth"
)

print("Fine-tuning completed!")
print("Average loss:", total_loss / max_batches)


import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio

# Test dataset
TEST_ROOT = "/kaggle/working/data/Flare7Kpp/test_data/synthetic"

to_tensor = transforms.ToTensor()

input_maes = []
output_maes = []
input_psnrs = []
output_psnrs = []

# Make sure fine-tuned weights are being evaluated
model.load_state_dict(
    torch.load(
        "/kaggle/working/flare_finetuned_500.pth",
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

for i in range(100):

    name = f"{i:06d}.png"

    input_img = Image.open(
        os.path.join(TEST_ROOT, "input", f"input_{name}")
    ).convert("RGB")

    gt_img = Image.open(
        os.path.join(TEST_ROOT, "gt", f"gt_{name}")
    ).convert("RGB")

    # Resize to training resolution
    input_256 = input_img.resize((256, 256))
    gt_256 = gt_img.resize((256, 256))

    x = to_tensor(input_256).unsqueeze(0).to(device)

    gt = np.array(gt_256).astype(np.float32) / 255.0
    original = np.array(input_256).astype(np.float32) / 255.0

    with torch.no_grad():
        pred = model(x)

    pred = pred.squeeze(0).cpu().clamp(0, 1)
    pred = pred.permute(1, 2, 0).numpy()

    input_maes.append(np.mean(np.abs(original - gt)))
    output_maes.append(np.mean(np.abs(pred - gt)))

    input_psnrs.append(
        peak_signal_noise_ratio(gt, original, data_range=1.0)
    )

    output_psnrs.append(
        peak_signal_noise_ratio(gt, pred, data_range=1.0)
    )

print("\n========== FINE-TUNED MODEL RESULTS ==========")

print("Average Input MAE:", np.mean(input_maes))
print("Average Model MAE:", np.mean(output_maes))

print("Average Input PSNR:", np.mean(input_psnrs))
print("Average Model PSNR:", np.mean(output_psnrs))

print(
    "\nImages improved by model:",
    sum(o < i for o, i in zip(output_maes, input_maes)),
    "/ 100"
)

print("\n========== COMPARISON WITH PREVIOUS MODEL ==========")

print("Previous Model MAE: 0.049669")
print("Current Model MAE:", round(np.mean(output_maes), 6))

print("Previous Model PSNR: 22.339 dB")
print("Current Model PSNR:", round(np.mean(output_psnrs), 4), "dB")


import shutil
import os

source = "/kaggle/working/flare_finetuned_500.pth"
backup = "/kaggle/working/flare_finetuned_500_backup.pth"

shutil.copy2(source, backup)

print("Fine-tuned checkpoint backed up!")
print("File size:", round(os.path.getsize(backup) / (1024**2), 2), "MB")


import torch
import torch.nn as nn

model.load_state_dict(
    torch.load(
        "/kaggle/working/flare_finetuned_500_backup.pth",
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-6
)

criterion = nn.L1Loss()

total_loss = 0
max_batches = 500

for batch_idx, (base, flare, merged, mask, gamma) in enumerate(train_loader):

    if batch_idx >= max_batches:
        break

    inputs = merged.to(device)
    targets = base.to(device)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    if (batch_idx + 1) % 100 == 0:
        print(f"Batch {batch_idx + 1}/500 | Loss: {loss.item():.6f}")

model.eval()

torch.save(
    model.state_dict(),
    "/kaggle/working/flare_finetuned_1000.pth"
)

print("Experiment completed!")
print("Average loss:", total_loss / max_batches)


import os
import glob

for file in glob.glob("/kaggle/working/**/*.pth", recursive=True):
    print(file)


import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
def load_model(checkpoint_path):
    model = UNet().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Support both checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

# Load all three models
original_model = load_model(
    "/kaggle/working/Flare_Removal_Project/checkpoints/baseline_256/baseline_unet_256_best.pth"
)

model_1 = load_model(
    "/kaggle/working/flare_finetuned_500.pth"
)

model_2 = load_model(
    "/kaggle/working/flare_finetuned_1000.pth"
)

print("All three models loaded successfully!")



import torch
import torch.nn as nn

# Two convolution layers with Batch Normalization and ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.final(d1))


# Verify architecture
model_check = UNet()

print("U-Net architecture restored!")
print("Parameters:", sum(p.numel() for p in model_check.parameters()))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    model = UNet().to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

original_model = load_model(
    "/kaggle/working/Flare_Removal_Project/checkpoints/baseline_256/baseline_unet_256_best.pth"
)

model_1 = load_model(
    "/kaggle/working/flare_finetuned_500.pth"
)

model_2 = load_model(
    "/kaggle/working/flare_finetuned_1000.pth"
)

print("All three models loaded successfully!")


import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

root = "/kaggle/working/data/Flare7Kpp/test_data/synthetic"
input_dir = os.path.join(root, "input")
gt_dir = os.path.join(root, "gt")

files = sorted(glob.glob(input_dir + "/input_*.png"))

random.seed(42)
selected = random.sample(files, 5)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict(model, path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)

    return output.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

def show_img(path):
    return Image.open(path).convert("RGB").resize((256, 256))

fig, axes = plt.subplots(5, 5, figsize=(20, 20))

for i, path in enumerate(selected):

    number = os.path.basename(path).replace("input_", "")
    gt_path = os.path.join(gt_dir, "gt_" + number)

    images = [
        show_img(path),
        predict(original_model, path),
        predict(model_1, path),
        predict(model_2, path),
        show_img(gt_path)
    ]

    titles = [
        "Input Image",
        "Original U-Net",
        "Experiment 1",
        "Experiment 2",
        "Ground Truth"
    ]

    for j in range(5):
        axes[i, j].imshow(images[j])
        axes[i, j].set_title(titles[j])
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()


def ensemble_predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_original = original_model(x)
        pred_exp1 = model_1(x)
        pred_exp2 = model_2(x)

        # Average predictions from all three models
        combined = (
            pred_original + pred_exp1 + pred_exp2
        ) / 3.0

    return combined.squeeze(0).cpu().permute(1, 2, 0).numpy().clip(0, 1)

print("Combined ensemble prediction function created!")


import os
import matplotlib.pyplot as plt

# Create a 5 x 6 comparison grid
fig, axes = plt.subplots(5, 6, figsize=(24, 20))

for i, path in enumerate(selected):

    # Find matching ground truth
    number = os.path.basename(path).replace("input_", "")
    gt_path = os.path.join(gt_dir, "gt_" + number)

    # Get predictions from all models
    images = [
        show_img(path),
        predict(original_model, path),
        predict(model_1, path),
        predict(model_2, path),
        ensemble_predict(path),
        show_img(gt_path)
    ]

    titles = [
        "Input Image",
        "Original U-Net",
        "Experiment 1",
        "Experiment 2",
        "Combined Ensemble",
        "Ground Truth"
    ]

    for j in range(6):
        axes[i, j].imshow(images[j])
        axes[i, j].set_title(titles[j], fontsize=11)
        axes[i, j].axis("off")

plt.suptitle(
    "Flare Removal - Individual Models vs Combined Ensemble",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


get_ipython().getoutput("git clone https://github.com/fida22/nighttime-flare-removal.git")


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("ls")


get_ipython().getoutput("git status")


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("find notebooks src docs results -maxdepth 2 -type f")


get_ipython().getoutput("find /kaggle/working/Flare_Removal_Project -maxdepth 3 -type f")


get_ipython().getoutput("ls -lah /kaggle/working")


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("cat .gitignore")


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("ls -lah src/training/")
get_ipython().getoutput("ls -lah notebooks/")


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("cp /kaggle/working/data_loader.py src/training/data_loader.py")

get_ipython().getoutput("ls -lh src/training/data_loader.py")


get_ipython().getoutput("find /kaggle/working -maxdepth 3 -type f \")
  \( -name "*.ipynb" -o -name "*.pdf" -o -name "*.docx" \) \
  ! -path "*/nighttime-flare-removal/*"


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("git status --short")


get_ipython().run_line_magic("cd", " /kaggle/working/nighttime-flare-removal")

get_ipython().getoutput("cp /kaggle/working/.virtual_documents/__notebook_source__.ipynb \")
  notebooks/leela_flare_removal_experiments.ipynb

get_ipython().getoutput("ls -lh notebooks/leela_flare_removal_experiments.ipynb")
