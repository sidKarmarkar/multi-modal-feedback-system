# Rich Feedback System for Text-to-Image Generation  
**Developed by**:  
- **Siddharth Hemant Karmarkar** (UNI: shk2195)  
- **Mihir Trivedi** (UNI: mpt2142)  

---

## Project Overview  

This project focuses on **Multi-Modal Evaluation** for refining **Text-to-Image Generation** models. It provides actionable feedback through multiple metrics, identifying artifact regions, misalignments, and producing fine-grained scores (Plausibility, Aesthetics, Alignment, and Overall Quality).  

Key features include:  
1. **Model**: Uses Vision Transformer (ViT), BERT, and T5 models for text and image evaluation.  
2. **Dataset**: RichHF-18K with fine-grained annotations on 18K images.  
3. **Evaluation**: Produces detailed feedback on generated images with metrics like **PLCC** and **SRCC**.  

---

## Directory Structure  

```
├── data  
│   ├── dataset.py               # Dataset handling module  
│   ├── train.tfrecord           # Training data in TFRecord format  
│   ├── dev.tfrecord             # Development data  
│   ├── test.tfrecord            # Test data  
│   ├── README.md                # Data instructions  
│   └── .gitattributes  
├── eecsenv                      # Virtual environment directory  
├── notebooks                    # Jupyter Notebooks for experiments  
├── src  
│   ├── __init__.py  
│   ├── feedback.py              # Core feedback model  
│   ├── model.py                 # Model architecture  
│   ├── README.md                # Code instructions  
│   ├── requirements.txt         # Dependencies  
│   └── __pycache__              # Cached Python files  
└── EECS6694_Slides_shk2195_mpt2142.pdf  # Supporting presentation  
```

---

## Prerequisites  

The following libraries and tools are required to run the project:  
- Python >= 3.8  
- PyTorch  
- TensorFlow  
- Transformers  
- Datasets  
- Torchvision  
- PIL (Pillow)  
- Requests  

---

## Installation  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/sidKarmarkar/multi-modal-feedback-system
   cd rich-feedback-system
   ```

2. **Set Up a Virtual Environment**  
   ```bash
   python3 -m venv eecsenv
   source eecsenv/bin/activate      # For MacOS/Linux
   eecsenv\Scripts\activate         # For Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r src/requirements.txt
   ```

---

## How to Run the Code  

1. **Run the Model**  
   ```bash
   python src/feedback.py
   ```

   - This script processes images and text prompts, returning plausibility, alignment, aesthetic scores, and overall quality.  
   - Example outputs are provided in the comments.  

2. **Run the Dataset Loader**  
   Modify the dataset path in `data/dataset.py` and execute:  
   ```bash
   python data/dataset.py
   ```

   - Ensure you replace `/path/to/your/data/train.tfrecord` with the actual dataset path.  

---

## Example Outputs  

For an input image-text pair:  
**Prompt**: *"A picnic scene with two people sitting and one standing"*  

**Output Scores**:  
- Plausibility: **0.4466**  
- Alignment: **0.5134**  
- Aesthetics: **0.5088**  
- Overall Quality: **0.4638**  

---

## Dataset  

The **RichHF-18K Dataset** contains:  
- Point annotations for artifacts and misalignments.  
- Misaligned keywords and fine-grained scores (Plausibility, Aesthetics, Alignment, and Overall Quality).  

### Example Record:  
```python
{
    'image': <image_tensor>,  
    'prompt': "A cat sitting on a table",  
    'aesthetics_score': 0.85,  
    'artifact_score': 0.12,  
    'misalignment_score': 0.3,  
    'overall_score': 0.8  
}
```

---

## Supporting Presentation  

Refer to **EECS6694_Slides_shk2195_mpt2142.pdf** for a detailed explanation of the:  
1. Project Motivation  
2. Model Architecture  
3. Dataset Contributions  
4. Evaluation Metrics  
5. Results and Practical Applications  

---

## Acknowledgments  

This project is a part of the **Generative AI and Modern Deep Learning** course (EECS6694) at Columbia University.

---
