import sagemaker
from sagemaker.pytorch import PyTorchModel

# AWS setup
role = ''
s3_model_path = ''

# Define the PyTorch model
pytorch_model = PyTorchModel(
    model_data=s3_model_path,
    role=role,
    entry_point='src/inference.py',  # Path to your inference script
    framework_version='1.9.0',  # Adjust to your PyTorch version
    py_version='py38'  # Adjust to your Python version
)

# Deploy the model
predictor = pytorch_model.deploy(
    instance_type='ml.m5.large',  # Adjust based on your needs
    endpoint_name='pytorch-endpoint',
    initial_instance_count=1
)
print("Model deployed to endpoint 'pytorch-endpoint'")
