import os
import sys
import torch
import logging
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JPEG_CONTENT_TYPE = 'image/jpeg'


def predict_fn(input_object, model):
    image = Image.open(BytesIO(input_object))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    processed_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(processed_image)
    return prediction


def input_fn(request_body, content_type):
    if content_type != JPEG_CONTENT_TYPE:
        raise Exception(f'Unsupported ContentType -> {content_type}')

    return request_body


def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 133))

    return model


def model_fn(model_dir):

    model = net()
    model.to(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)

    model.eval()
    return model
