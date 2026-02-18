# LoRA_Assistant
Project for training a LoRA model to assist a user in geographic knowledge

# Setting up

Use conda to setup an environment: 

```conda create -n training python=3.11 -y conda activate training```

Make sure to install the required packages:

```pip install --upgrade pip```

```pip install torch --index-url https://download.pytorch.org/whl/cu121```

```pip install clearml datasets```

```pip install "transformers>=4.45" accelerate bitsandbytes safetensors sentencepiece```

Note: in the case of CUDA errors with the setup, try the following:
``` pip uninstall torch torchvision torchaudio -y
 
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

# Model Training

The script `train_model.py` performs the training using the data in the `training_data` folder

## LoRA Training Configuration (PEFT) — Parameter Reference

The main configuration parameters commonly found in a **LoRA (Low-Rank Adaptation)** training setup using the **Hugging Face PEFT library** are explained below:

The example configuration below is designed for fine-tuning the model:

- **Base model:** `microsoft/Phi-3.5-mini-instruct`  
- **Task:** Causal Language Modeling  
- **Adapter method:** LoRA
