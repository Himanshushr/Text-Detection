cat << EOF > README.txt
# 📖 ViT-CTC Image Sequence Recognition

This file documents the PyTorch code for training and inference using a 
Vision Transformer (ViT) integrated with the Connectionist Temporal 
Classification (CTC) Loss for image sequence recognition (e.g., OCR).

The code is optimized for execution in a **Google Colab** environment.

---

## 🛠️ Project Components

| File/Section | Description |
| :--- | :--- |
| ViT_CTC_Model | PyTorch module combining a pre-trained ViT backbone with a linear layer for CTC output. |
| RecognitionDataset | Custom PyTorch Dataset class for loading images (.png) and their text labels (.txt). |
| Training Script | Handles data loading, loss calculation (CTCLoss), and model optimization. |
| Inference Script | Loads the saved model and implements Greedy CTC Decoding to convert logits to text. |

## ⚙️ Setup and Prerequisites

### 1. Environment
The code is designed to run in **Google Colab** using the available GPU (CUDA).

### 2. Dependencies
The following packages are required:
\`\`\`bash
!pip install transformers torch pillow torchvision
\`\`\`

### 3. Data Structure **(Crucial Constraint)**

The script requires the dataset to be organized into the following structure in the Colab root directory:

/recognition_dataset
├── /images
│   ├── 001.png
│   └── ...
└── /labels
    ├── 001.txt    <- Must contain the ground truth text for 001.png
    └── ...

**Constraint:** Image and label files must share the same base filename (e.g., '123.png' must correspond to '123.txt').

---

## 🔑 Training Details and Constraints

### Model & Hyperparameters

| Parameter | Value / Description | Constraint/Context |
| :--- | :--- | :--- |
| **Backbone** | google/vit-base-patch16-224 | Pre-trained model. Input images are resized to 224x224 pixels. |
| **Output Seq. Len.** | 197 | Derived from ViT patch size: (224/16)^2 + 1 (for the [CLS] token). |
| **Loss Function** | torch.nn.CTCLoss(blank=0) | **Index 0 is reserved for the blank token.** |
| **Character Set** | 63 classes | Includes 0-9, a-z, A-Z, plus the one required <blank> token (index 0). |

### Execution Constraints

1.  **Data Consistency:** All characters in the .txt labels **must** be present in the defined \`charset\`. Characters outside this set will be ignored.
2.  **Model Saving:** The trained model weights are saved to a file named \`vit_ctc.pth\`.

---

## ▶️ Inference Process and Decoding

The decoding step converts the model's sequence of logits into the final text string.

### Decoding Constraint
* **Method:** **Greedy Decoding** is used. This involves taking the argmax (most probable character) at every time step and applying two post-processing rules:
    1.  Removing all occurrences of the **blank token** (index 0).
    2.  Collapsing consecutive **duplicate** non-blank characters.

### Usage
The inference script automatically loads the \`vit_ctc.pth\` weights and tests the prediction on the first available image, displaying both the Ground Truth and the Predicted Label.

EOF
