import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def model_fn(model_dir):
    model = Generator()
    model_path = f'{model_dir}/model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_fn(input_data, model):
    num_images = input_data.get('num_images', 1)
    noise = torch.randn(num_images, 100, 1, 1)
    with torch.no_grad():
        generated_images = model(noise)
    return generated_images
