import os
import sys
import torch
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    logger.info("Testing started")
    test_loss = correct = 0

    model.eval()

    with torch.no_grad():
        for (data, target) in test_loader:
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            loss = criterion(outputs, target)
            test_loss += loss.item()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {correct / len(test_loader.dataset)}")

    logger.info("Testing completed")


def train(model,
          train_loader,
          validation_loader,
          criterion,
          optimizer,
          epochs,
          early_stopping,
          device):

    logger.info("Training started")

    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': validation_loader}
    loss_counter = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                if running_samples % 2000 == 0:
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0 * accuracy,
                        )
                    )

                if running_samples > (0.2 * len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter += 1
            logger.info(
                '{} Loss: {:.4f}, Accuracy: {:.4f}, Best loss: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, best_loss))

        if loss_counter == early_stopping:
            logger.info('Training early stopping')
            break

    logger.info("Training completed")
    return model


def net():
    logger.info("Model creation started")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 133))

    logger.info("Model creation completed")
    return model


def create_data_loaders(data_dir, batch_size):
    train_data_path = os.path.join(data_dir, "train")
    test_data_path = os.path.join(data_dir, "test")
    validation_data_path = os.path.join(data_dir, "valid")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.ImageFolder(
        root=train_data_path,
        transform=train_transform
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    test_data = datasets.ImageFolder(
        root=test_data_path,
        transform=test_transform
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    validation_data = datasets.ImageFolder(
        root=validation_data_path,
        transform=test_transform,
    )

    validation_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_data_loader, test_data_loader, validation_data_loader


def main(args):
    logger.info(f"Hyperparameters: {args}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=args.learning_rate)

    train_loader, test_loader, validation_loader = create_data_loaders(
        args.data_dir, args.batch_size)

    model = train(model,
                  train_loader,
                  validation_loader,
                  criterion,
                  optimizer,
                  args.epochs,
                  args.early_stopping_rounds,
                  device)

    test(model, test_loader, criterion, device)

    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument('--early-stopping-rounds', type=int, default=10)
    parser.add_argument("--model-dir", type=str,
                        default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str,
                        default=os.environ["SM_CHANNEL_TRAINING"])

    args, _ = parser.parse_known_args()

    main(args)
