# AWS Sagemaker Inference Lambda
This repository is directly linked with Serverless-Object-Detection-Web-Application. The lambda function in this repositoy requires `sagemaker` package which is not supported by AWS Lambda, so it will be packaged with the lambda handler and deployed as a container. This container image is stored on AWS ECR. See Dockerfile for containerization details.
