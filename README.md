# SynTeX: Efficient LaTeX OCR with Synthetic Pretraining

SynTeX is a data-efficient LaTeX OCR system that converts scientific document images into editable LaTeX code. By introducing a novel synthetic pretraining approach that pairs Wikipedia text with LaTeX formulas, SynTeX achieves competitive performance with only **400 fine-tuning samples**, compared to existing methods requiring millions of real paired samples.

## Overview

**Key Features:**
- **Data Efficient**: Only 400 fine-tuning samples needed (vs. millions for baselines)
- **Cross-lingual**: Supports both English and Chinese scientific documents
- **Multi-format**: Handles printed and handwritten content
- **Synthetic Pretraining**: 120k synthetic pages from Wikipedia + 2M LaTeX formulas
- **Open Source**: Models, code, and datasets fully released

**Architecture:**
- **Encoder**: Swin Transformer (base size)
- **Decoder**: GPT-2 (medium size)
- **Training**: Two-stage pipeline (synthetic pretraining → real data fine-tuning)


**Test Datasets:**
| Dataset | Images | Language | Type | Size |
|---------|--------|----------|------|------|
| data_printed_chinese | 461 | Chinese | Printed | 7.8M |
| data_printed_english | 367 | English | Printed | 3.5M |
| data_handwritten_english | 81 | English | Handwritten | 2.5M |
| data_handwritten_chinese | 68 | Chinese | Handwritten | 1.3M |
| **Total** | **1,018** | - | - | **15.1M** |

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/syntex.git
cd syntex

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.12
- Transformers >= 4.20
- CUDA >= 11.0 (for GPU acceleration)

### Inference (Python)

```python
from code.inference import SynTeXInference

# Initialize model (from local files)
inferencer = SynTeXInference(model_path="./model")

# Run inference on a single image
latex_code = inferencer.predict("path/to/image.jpg")
print(latex_code)
```

### Inference (Command Line)

```bash
# Single image
python code/inference.py --model ./model --image path/to/image.jpg

# Batch processing directory
python code/inference.py --model ./model --image_dir ./test_data/ --output results.json
```

### Evaluation

```bash
python code/evaluate.py \
    --model ./model \
    --csv path/to/ground_truth.csv \
    --image_dir path/to/images/ \
    --output evaluation_results.json
```

## Datasets

### Pretraining Data (Synthetic)

**Location:** `./pretrain_data/`

**Contents:**
- `en1.zip`, `en2.zip`, `en3.zip` - English Wikipedia text + LaTeX formulas
- `zh1.zip`, `zh2.zip`, `zh3.zip` - Chinese Wikipedia text + LaTeX formulas
- `./pretrain_data_gen_code/` - Data generation scripts

**Total:** ~120,000 synthetic document pages

**Format:**
```
pretrain_data/
├── en1.zip
├── en2.zip
├── en3.zip
├── zh1.zip
├── zh2.zip
└── zh3.zip
```

### Test Data (Real Documents)

**Location:** `./test_data/`

**Dataset Format (each zip):**
```
data_X_*.zip
├── *.jpg (images)
└── ground_truth.csv  # Format: file_name,text
```

**ground_truth.csv Example:**
```csv
file_name,text
1724718624.jpg,"..."
```

**Data Source:** Scientific documents from arXiv and Chinese academic repositories

For more details, see `datasets/dataset_description.md`

## Model

**Location:** `./model/`

**Contents:**
- Model weights in HuggingFace style
- Configuration files
- Tokenizer files

**Load from local:**
```python
from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor

feature_extractor = AutoImageProcessor.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model", max_length=296)
model = VisionEncoderDecoderModel.from_pretrained("./model")
```

## Training

### Train with HuggingFace Dataset

```bash
python code/train.py --mode huggingface --output_dir ./results
```

### Train with Local CSV Data

```bash
python code/train.py \
    --mode local \
    --csv path/to/ground_truth.csv \
    --image_dir path/to/images/ \
    --output_dir ./results
```

## File Structure

```
syntex/
├── README.md                      # This file
├── requirements.txt               # Python dependencies│
├── code/                          # Source code
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference script
│   ├── evaluate.py               # Evaluation script
│   └── utils.py                  # Utility functions
│
├── model/                         # Pre-trained model weights
│   ├── pytorch_model.bin         # Model weights
│   ├── config.json               # Model configuration
│   ├── tokenizer_config.json     # Tokenizer configuration
│   └── ...                       # Other model files
│
├── test_data/                     # Test datasets
│   ├── data_printed_chinese.zip
│   ├── data_printed_english.zip
│   ├── data_handwritten_english.zip
│   └── data_handwritten_chinese.zip
│
├── pretrain_data/                 # Synthetic pretraining data
│   ├── en1.zip
│   ├── en2.zip
│   ├── en3.zip
│   ├── zh1.zip
│   ├── zh2.zip
│   └── zh3.zip
│
├── pretrain_data_gen_code/        # Data generation code
│   ├── gen.py
│   ├── formular.tex
│   └── endata1.txt

**Note:** This is the official implementation of SynTeX. If you find this code useful, please consider starring ⭐ our repository!
