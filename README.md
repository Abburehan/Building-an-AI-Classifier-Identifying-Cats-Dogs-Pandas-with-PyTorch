# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch

This project builds a deep learning image classifier that identifies **cats**, **dogs**, and **pandas** using **PyTorch** and **Transfer Learning**. A pre-trained **ResNet** model is fine-tuned on a custom dataset with full support for both **local GPU**, **CPU**, and **Kaggle notebook** execution.

---

## Features

- ✔ Transfer Learning with a pre-trained ResNet architecture  
- ✔ Data augmentation & preprocessing using `torchvision.transforms`  
- ✔ Structured training, validation & testing loops  
- ✔ Visualization of training/validation loss & accuracy  
- ✔ Automatic GPU (CUDA) detection and usage  
- ✔ Works seamlessly on both local machines and Kaggle  

---

## Project Structure

DL_Classification_Project/
│

├── DL_Classification_project.ipynb 

├── README.md
│
└── dataset/

├── train/

├── valid/

└── test/

```
pip install torch torchvision torchaudio matplotlib numpy pandas tqdm
```

## How to Use the Notebook
## 1️⃣ Open the Notebook

```
jupyter notebook DL_Classification_project.ipynb
```
## 2️⃣ Preparing the Dataset

Download any cats–dogs–pandas dataset (Kaggle recommended).
Arrange it in the following structure:

dataset/

 ├── train/
 
 ├── valid/
 
 └── test/

## 3️⃣ Verify CUDA (GPU) Availability

Inside the notebook, run:

```
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

If CUDA is unavailable, the code automatically switches to CPU.

## Running on Kaggle

1. Upload the notebook to Kaggle

2. Enable GPU:
   Settings → Accelerator → GPU (T4 or P100)

3. Attach dataset via Add Data → Cats and Dogs and Pandas Dataset

4. Set the dataset path:
   
```
data_dir = "/kaggle/input/cats-and-dogs-and-pandas/"
```

5. Run all cells — training, evaluation, and plots will execute automatically.

## Model Performance

🎯 Fine-tuned ResNet model

📈 High accuracy on test images

🖼 Clear plots of training & validation curves

🐱🐶🐼 Model successfully identifies all 3 classes

## Result

Thus an AI Classifier that Identifies Cats, Dogs & Pandas with PyTorch is built successfully.
