import boto3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# AWS setup
s3_bucket = 'vanillagan'
s3_prefix = 's3://'
aws_access_key_id = ''
aws_secret_access_key = ''
aws_region = 'us-east-1'

# PyTorch setup
batch_size = 64
image_size = 64
latent_dim = 100
num_epochs = 5
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom dataset class to load images from S3
class S3ImageDataset(Dataset):
    def __init__(self, s3_bucket, s3_prefix, transform=None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform

        # Initialize the S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # List all files in the S3 bucket/prefix
        self.file_list = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.s3_bucket)

        for page in pages:
            for item in page.get('Contents', []):
                self.file_list.append(item['Key'])
        print(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_key = self.file_list[idx]
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=file_key)
        img = Image.open(BytesIO(response['Body'].read())).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img

# Define the transformation and load the dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = S3ImageDataset(s3_bucket, s3_prefix, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
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

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Initialize the models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        real_images = data.to(device)
        batch_size = real_images.size(0)
        labels = torch.full((batch_size,), 1., dtype=torch.float, device=device)
        output = netD(real_images).view(-1)
        lossD_real = criterion(output, labels)
        lossD_real.backward()
        
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        labels.fill_(0.)
        output = netD(fake_images.detach()).view(-1)
        lossD_fake = criterion(output, labels)
        lossD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labels.fill_(1.)
        output = netD(fake_images).view(-1)
        lossG = criterion(output, labels)
        lossG.backward()
        optimizerG.step()

        print(f'Epoch [{epoch}/{num_epochs}] Step [{i}/{len(dataloader)}] '
              f'Loss D: {lossD_real.item() + lossD_fake.item()}, Loss G: {lossG.item()}')

# Inference: Generate images
def generate_images(num_images):
    netG.eval()
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        generated_images = netG(noise).cpu()
    return generated_images

# Generate a batch of images
num_images_to_generate = 16
generated_images = generate_images(num_images_to_generate)

# Function to show images
def show_images(images):
    grid = np.transpose(torchvision.utils.make_grid(images, padding=2, normalize=True), (1, 2, 0))
    plt.imshow(grid)
    plt.show()

# Display the generated images
show_images(generated_images)
