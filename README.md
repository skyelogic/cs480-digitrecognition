# TensorFlow.js Digit Recognition Demo

---

## 👨‍💻 Author
**Donnel Garner**  
Old Dominion University  
Norfolk, Virginia  

**CS480 - Introduction to Artificial Intelligence**  
**Spring 2026**

📅 **Due Date:** March 3rd, 2026
🔗 **GitHub Repository:** https://github.com/skyelogic  
🌐 **Live Demo:** https://donnelgarner.com/projects/CS480/digit-recognition/

---

## Overview
This project trains a multi-layer perceptron (MLP) neural network to recognize
handwritten digits (0-9) using 32x32 binary bitmap images from the UCI optdigits
dataset. It covers:

- **Task 1**: Parsing and encoding the raw bitmap data into CSV input/target files
- **Task 2**: Training MLP neural networks with 10, 100, and 500 hidden nodes
- **Bonus**: PCA dimensionality reduction to reduce the 1024-feature input space

---

## 📋 Table of Contents
- [How to Run](#-how-to-run)
- [Task 1: Data Encoding](#-task-1-data-encoding)
- [Task 2: Neural Network Training](#-task-2-neural-network-training)
- [Task 3: Side-by-Side Model Comparison](#-task-3-side-by-side-model-comparison)
- [Bonus: PCA Dimensionality Reduction](#-bonus-pca-dimensionality-reduction)
- [Technologies Used](#️-technologies-used)
- [References](#-references)

---

# 🚀 How to Run

## Requirements

Install dependencies:
    pip install numpy scikit-learn

## Data

Decompress the data file first:
    uncompress optdigits-orig.windep.Z

This produces optdigits-orig.windep (the raw bitmap file).

The script also prints accuracy (training, validation, test) for each hidden
layer size, per-class precision/recall/F1, summary table, and PCA bonus results.

### Option 1: Python Local Server (Recommended)

    python encode_and_train.py --data optdigits-orig.windep

Optional flags:
    --data     Path to the .windep data file  (default: optdigits-orig.windep)
    --out_dir  Directory to save CSV outputs  (default: encoded_data/)

## Output Files

After running, the following are generated:

    encoded_data/
        inputs.csv   -- 1797 rows x 1024 columns (flattened bitmap pixels, 0 or 1)
        targets.csv  -- 1797 rows x 1 column     (digit label 0-9)

### Option 2: Python Local Server (Recommended)

```bash
cd digit-demo/
python -m http.server 8000
```

Then open:

```
http://localhost:8000/optdigits_side_by_side.html
```

---

### Option 3: Open Directly in Browser

Simply open the HTML file:

```
optdigits_side_by_side.html
```

⚠️ Some browsers block certain APIs when opened directly from disk, so using a local server is recommended.

---

### How to Use the Demo

1️⃣ **Draw a number with your mouse.**  
Try drawing **thicker lines** — TensorFlow appreciates bold artistic expression.

2️⃣ **Click "Train".**  
This trains three neural networks simultaneously.

3️⃣ **View Results.**  
The models will attempt to identify your masterpiece.

4️⃣ **Want to try again?**  
Refresh the page. Neural networks have short memories.

---

# 📊 Task 1: Data Encoding

The original **UCI Optdigits dataset** contains handwritten digits stored as:

- 32 rows
- 32 columns
- Binary pixel values (0 or 1)

Each digit is converted into a **1024-feature vector**.

| Dataset Property | Value |
|------------------|------|
| Image Size | 32 × 32 |
| Features | 1024 |
| Classes | 10 digits (0-9) |
| Total Samples | 1797 |

Encoded outputs:

```
encoded_data/
    inputs.csv
    targets.csv
```

| File | Description |
|-----|-------------|
| inputs.csv | 1797 × 1024 flattened pixel matrix |
| targets.csv | digit label (0-9) |

---

# 🧠 Task 2: Neural Network Training

Three neural networks were trained using **scikit-learn's MLPClassifier**.

| Model | Hidden Nodes |
|------|-------------|
| Small Model | 10 |
| Medium Model | 100 |
| Large Model | 500 |

Architecture:

| Layer | Size |
|------|------|
| Input | 1024 nodes |
| Hidden | 10 / 100 / 500 |
| Output | 10 nodes (digit classes) |

Training details:

- Activation: ReLU
- Optimizer: Adam
- Early stopping: enabled
- Max iterations: 500
- Data split: 60% train / 20% validation / 20% test

---

# 🔬 Task 3: Side-by-Side Model Comparison

The interactive demo trains three models simultaneously and shows how model size impacts predictions.

| Model | Behavior |
|------|----------|
| 10 nodes | Fast but sometimes confused |
| 100 nodes | Balanced accuracy |
| 500 nodes | More expressive but slower |

The interface displays:

- predicted digit
- probability/confidence bars
- hidden neuron activation info

This visual comparison demonstrates how increasing network capacity affects classification performance.

---

# 🎯 Bonus: PCA Dimensionality Reduction

The original input space contains **1024 features**.

Principal Component Analysis (PCA) reduces dimensionality while preserving useful information.

| PCA Components | Result |
|----------------|-------|
| 20 | moderate accuracy |
| 50 | strong performance |
| 100 | very high variance retained |
| 150–200 | near full representation |

Observation:

📊 Around **~100 PCA components retain most useful information**  
⚡ Training becomes significantly faster  
💾 Memory usage decreases

---

# 🛠️ Technologies Used

### Languages
- Python
- JavaScript
- HTML5
- CSS3

### Machine Learning Libraries
- scikit-learn
- NumPy
- TensorFlow.js

### Tools
- Visual Studio Code
- GitHub
- Python 3
- Terminal / SSH

### Browser APIs
- Canvas API (digit drawing)
- TensorFlow.js WebGL backend
- DOM manipulation
- Event listeners

---

# 📚 References

- Russell & Norvig — *Artificial Intelligence: A Modern Approach*
- UCI Machine Learning Repository – Optdigits dataset
- TensorFlow.js Documentation
- scikit-learn Documentation
- Visual Studio Code
- GitHub

---

# 📝 License

This project is submitted as coursework for **CS480 – Artificial Intelligence** at **Old Dominion University**.

---

# 🙏 Acknowledgments

Special thanks to:

- **Dr. Vikas Ashok** — Course Instructor
- **Old Dominion University** — Computer Science Department

---

<p align="center">
<strong>Made with ☕ and 💻 by Donnel Garner</strong><br>
<sub>Old Dominion University | CS480 | Spring 2026</sub>
</p>

---

<p align="center">
<a href="https://donnelgarner.com">🌐 Personal Website</a> •
<a href="https://github.com/skyelogic">💻 GitHub</a>
</p>
