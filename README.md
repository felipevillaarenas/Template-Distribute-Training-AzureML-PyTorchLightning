# Template-Distribute-Training-AzureML-PyTorchLightning
Template for Distributed training on AzureML with Pytorch Lightning

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.0+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://ml.azure.com/"><img alt="MLAzure" src="https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white"></a>

## 📌&nbsp;&nbsp;Introduction
This repository provides a PyTorch Lighting Template for Distribute Training on Azure ML.

> This template was aim to user PytorchLightning 2.0 or Higher. 



### Why PyTorch Lightning?
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a lightweight PyTorch wrapper for high-performance AI research.
It makes your code neatly organized and provides lots of useful features, like ability to run model on CPU, GPU, multi-GPU cluster and TPU.

### Why Azure ML?
It comes to the partnership between Microsoft Azure and OpenAI. This groundbreaking collaboration introduces a cloud-based platform designed to empower developers and data scientists to swiftly and effortlessly build and deploy AI models. Leveraging Azure OpenAI, users gain access to a comprehensive suite of cutting-edge AI tools and technologies, enabling intelligent applications that harness the power of natural language processing, computer vision, and deep learning.

### Code Structure
To adhere to best practices, we will store all Azure SDK-related code in separate Python files located in the azure-jobs folder. Jobs can be seen as the connecting element between the compute cluster, data asset components, and PyTorch code. Conversely, the native PyTorch code will be placed in the src folder. As a result, if we decide to run our training on a different cloud provider, no modifications will be required in the src folder.
Here is an example of how folder extractor could look like:
```
.
├── README.md
├── azure-jobs/
│   ├── config/
│   │   └── workspace.json
│   └── job.py
└── src/
    ├── datamodule.py
    ├── model.py
    ├── transforms.py
    └── trainer.py
```