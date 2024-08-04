FROM pytorch/pytorch:1.8.1-cpu

RUN pip install boto3 torchvision matplotlib pillow sagemaker

COPY train.py /opt/ml/code/train.py
COPY inference.py /opt/ml/code/inference.py
WORKDIR /opt/ml/code
ENTRYPOINT ["python", "train.py"]
