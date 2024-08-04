import sagemaker
from sagemaker.predictor import Predictor
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# AWS setup
endpoint_name = 'pytorch-endpoint'  # The name of the endpoint you deployed

# Initialize a Predictor
predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker.Session(),timeout=600)

def generate_and_save_images(num_images):
    # Prepare the input data for the prediction
    input_data = {'num_images': num_images}
    
    # Serialize input data to JSON and then encode as bytes
    input_data_json = json.dumps(input_data)
    
    # Send the data to the endpoint and get predictions
    response = predictor.predict(input_data_json.encode('utf-8'))
    
    # Convert response to images
    # Assuming response is in the form of a list of images in bytes
    generated_images = torch.tensor(np.frombuffer(response, dtype=np.float32).reshape(-1, 3, 64, 64))  # Adjust dimensions as necessary
    
    # Save images
    transform = transforms.ToPILImage()
    for i in range(num_images):
        img = transform(generated_images[i])
        img.save(f'generated_image_{i}.png')

# Example usage
num_images_to_generate = 16
generate_and_save_images(num_images_to_generate)
print("Images generated and saved successfully")
