import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_device
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time

from dataset import POCDataset
from model import ResNet18


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc


def main():
    device = get_device()

    data_dir = "./POC_Dataset"
    num_classes = 4
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-4

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(), # data augmentation
        T.RandomRotation(10), # data augmentation
        T.ToTensor(),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # datasets (Training 폴더에 있는 데이터 -> train/val split)
    train_dataset = POCDataset(
        data_dir=data_dir,
        data_type="Training",
        transform=train_transform,
        split="train"
    )

    val_dataset = POCDataset(
        data_dir=data_dir,
        data_type="Training",
        transform=val_transform,
        split="val"
    )

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    # model, loss, optimizer
    model = ResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 5 #early stopping patience
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch+1}/{num_epochs}] ({elapsed:.2f} sec)")
        print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print("-" * 50)

        # log 기록
        with open("./result/training_log.txt", "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f},{elapsed:.2f}\n")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            early_stop_counter = 0

            torch.save(model.state_dict(), "./result/best_resnet18_poc.pth")
            print(f"\n Best model saved -> Epoch {epoch+1}, Val Acc = {val_acc:.2f}%\n")
        else:
            early_stop_counter += 1

        # early stopping
        if early_stop_counter >= patience:
            print(f"\n Early stopping triggered at epoch {epoch+1}")
            print(f"Best epoch = {best_epoch}, Best Val Acc = {best_val_acc:.2f}%\n")
            break

    print(f"\nTraining Finished. Best Val Acc = {best_val_acc:.2f}% (Epoch {best_epoch})\n")

    # loss plot
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.xticks(range(len(train_losses)))
    plt.legend()
    plt.savefig("./result/loss_curve.png")
    plt.show()

    # accuracy plot
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.xticks(range(len(train_accs)))
    plt.legend()
    plt.savefig("./result/accuracy_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
