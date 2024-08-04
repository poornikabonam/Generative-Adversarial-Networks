import sagemaker
from sagemaker.pytorch import PyTorch

role = ''
sagemaker_session = sagemaker.Session()

# Define the PyTorch estimator
estimator = PyTorch(
    entry_point='src/train.py',
    source_dir='.',
    role=role,
    instance_type='ml.m4.2xlarge',
    instance_count=1,
    framework_version='1.8.1',
    py_version='py3',
    hyperparameters={
        'batch_size': 64,
        'num_epochs': 5,
        'latent_dim': 100,
        'lr': 0.0002,
        'beta1': 0.5
    }
)

# Start training
estimator.fit(inputs={'training': 's3://vanillagan'})
