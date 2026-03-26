"""
Inference script for NYCU Computer Vision 2026 HW1.
This script loads a trained ResNet101 model and generates predictions.
"""

import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnlabeledTestDataset(Dataset):
    """Custom dataset for loading unlabeled test images."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Filter for image files and sort them
        valid_extensions = ('.png', '.jpg', '.jpeg')
        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(valid_extensions)
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name


def main():
    # --- CONFIG ---
    data_folder = 'data'
    batch_size = 64
    epochs = 5
    

    # --- DATA LOADERS ---
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_folder, 'train')
    val_path = os.path.join(data_folder, 'val')
    test_path = os.path.join(data_folder, 'test')

    train_set = datasets.ImageFolder(train_path, train_tf)
    val_set = datasets.ImageFolder(val_path, val_tf)
    test_set = UnlabeledTestDataset(test_path, val_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # --- MODEL & OPTIMIZER ---
    # Using ResNet101 with updated weights API
    weights = models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 100)

    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("Warning: best_model.pth not found! Starting from ImageNet.")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(DEVICE.type)

    # --- TRAINING LOOP ---
    best_acc = 88.0
    
    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs}", 
            unit="batch"
        )
        
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast(DEVICE.type):
                output = model(imgs)
                loss = criterion(output, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with autocast(DEVICE.type):
                    output = model(imgs)
                
                pred = output.argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()
        print(f"Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")

    # --- FINAL SUBMISSION ---
    print("\nTraining complete. Generating prediction.csv...")
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model = model.to(DEVICE)    
    model.eval()
    
    idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}
    results, clean_filenames = [], []

    with torch.no_grad():
        for imgs, fnames in tqdm(test_loader, desc="Predicting"):
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            preds = output.argmax(dim=1).cpu().numpy()

            for p in preds:
                results.append(idx_to_class[p])
            for f in fnames:
                clean_filenames.append(os.path.splitext(f)[0])

    df = pd.DataFrame({'image_name': clean_filenames, 'pred_label': results})
    df.to_csv('prediction.csv', index=False)
    print("Done!")


if __name__ == '__main__':
    main()