import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import POCDataset
from model import ResNet18
from utils import evaluate_and_visualize, get_device


def main():
    device = get_device()

    data_dir = "./POC_Dataset"

    class_names = [
        "Chorionic_villi",
        "Decidual_tissue",
        "Hemorrhage",
        "Trophoblastic_tissue"
    ]

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    test_dataset = POCDataset(
        data_dir=data_dir,
        data_type="Testing",
        transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ResNet18(num_classes=4).to(device)
    model_path = "./result/best_resnet18_poc.pth" # Best Model Weights 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    evaluate_and_visualize(
        model=model,
        loader=test_loader,
        device=device,
        class_names=class_names,
        save_csv=True,                       
        csv_path="./result/test_predictions.csv",
        cm_path="./result/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
