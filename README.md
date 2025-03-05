# 🌟 Multilingual Sentiment Analysis using LLaMA 3.1-8B-Instruct

![Kaggle Badge](https://img.shields.io/badge/Kaggle-Rank%2032-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red?style=flat-square)

## ✨ Overview
This project focuses on **sentiment analysis** for **13 Indian languages**, handling long text sequences up to **4096 tokens**. We fine-tuned **LLaMA 3.1-8B-Instruct** using **LoRA (Low-Rank Adaptation)** to enhance efficiency and trained it under **compute constraints** on Kaggle. 

### 🔄 Key Achievements:
- Achieved an **F1 score of 0.97435**, ranking **#32** in the Kaggle competition.
- Optimized **4-bit precision training** to fit within Kaggle's hardware limits.
- Leveraged **instruction-based prompts** to enhance model accuracy.

---
## 💡 Features
- Supports **positive/negative** sentiment classification.
- Handles **long-text sequences** (up to 4096 tokens).
- Efficient **parameter-efficient fine-tuning** using **LoRA**.
- Optimized for **low-resource multilingual NLP**.
- **Lightweight inference** using **4-bit precision quantization**.

---
## 🛠️ Tech Stack
- **Model**: [LLaMA 3.1-8B-Instruct](https://huggingface.co/)
- **Fine-Tuning**: LoRA, PEFT (Parameter Efficient Fine-Tuning)
- **Libraries**: PyTorch, Hugging Face Transformers, Unsloth, BitsAndBytes
- **Compute**: Kaggle Notebooks

---
## 📚 Setup & Usage
### 1️⃣ Install Dependencies
```bash
pip install torch transformers accelerate peft bitsandbytes unsloth
```

### 2️⃣ Load the Fine-Tuned Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "path/to/your/fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
```

### 3️⃣ Perform Sentiment Analysis
```python
text = "राजनीतिक निर्णय की वास्तविक है।"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---
## 💪 Contributions
Feel free to contribute by submitting **issues**, **pull requests**, or suggesting **improvements**. Let's make multilingual NLP more accessible! 

---
## 🌐 References
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
- [BitsAndBytes Quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
