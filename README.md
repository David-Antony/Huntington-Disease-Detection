<h1 align="center">🧠 Quantum-Inspired Deep Learning for Huntington's Disease Detection</h1>

<p align="center">
  <em>A cutting-edge, PyTorch-based neural framework leveraging quantum computing principles for the early detection and structural analysis of Huntington's Disease through MRI brain scans.</em>
</p>

## 🔬 Project Overview

Huntington's Disease (HD) is a genetic, progressive neurodegenerative disorder. Early detection via neuroimaging (MRI) is critical but challenging due to the subtle variations in structural brain degradation during early stages. 

This project introduces a **Quantum-Inspired Neural Network (QINN)** that augments a standard convolutional backbone (`EfficientNet-B0`) with a custom stack of continuous-variable quantum layers. By applying mathematical concepts derived from quantum mechanics—Superposition, Entanglement, Interference, and Measurement—the network is able to extract and correlate highly complex, multi-dimensional geometric features that traditional neural layers often miss.

## 🌌 Quantum-Inspired Architecture

The core innovation is the `QuantumFeatureBlock`, exclusively implemented in differentiable PyTorch tensors, placed directly before the final classification head. It computes features through four distinct stages:

### 1. Quantum Superposition Layer
Simulates the ability of a quantum system to exist in multiple states simultaneously. The input tensor is mapped into multiple parallel pathways ("basis states"), which are then combined using learnable **amplitudes** and **phases** via quantum phase modulation (cosine/sine wave representation).

### 2. Quantum Entanglement Layer
Imitates quantum entanglement, where particles become fundamentally interconnected. We implement this using a localized **self-attention mechanism**. A correlation matrix is calculated among feature dimensions, allowing features to instantaneously influence one another based on an learned `entanglement_strength` parameter.

### 3. Quantum Interference Layer
Replicates constructive and destructive wave interference. Features are bifurcated into two conceptual "wave functions." By applying a learnable `phase_diff`, the waves either amplify critical disease-specific biomarkers (constructive interference) or cancel out noise and unrelated structures (destructive interference) using a trainable interference gate.

### 4. Quantum Measurement Layer
Uses the **Born rule** to simulate wave function collapse. We define learnable `basis_vectors` and compute probability overlaps. The superposition collapses into a definitive feature state probabilistically, constrained by a temperature sharpness parameter mapping to definitive classical outputs.

## 📊 Training Methodology & Performance

We utilized a structured progressive two-phase fine-tuning approach:
- **Phase 1 (Warmup):** The CNN Backbone is frozen; only the `QuantumFeatureBlock` and Classifier are trained using a higher learning rate scheduler (`CosineAnnealingLR`) to initialize the quantum logic gates.
- **Phase 2 (Fine-tuning):** The final layers of the `EfficientNet` backbone are unfrozen and trained jointly with a much lower, decoupled learning rate.

**Results:**
The framework successfully captures underlying structural abnormalities and converges to **>95.5% Accuracy** across training, validation, and hold-out test sets while preventing overfitting via strict label smoothing and dropout regulators.

## 🚀 User Interface & Deployment (Hugging Face Spaces)

A production-ready Web App is built alongside the core model, crafted natively for deployment on Hugging Face Spaces.

- **Backend Engine:** A lightweight Flask server loading the serialized PyTorch models asynchronously, processing image arrays via `/api/predict`.
- **Frontend Intelligence:** A visually striking interface built with Tailwind CSS. It highlights Modern Glassmorphism effects, interactive drag-and-drop zones, and CSS-driven particle background canvases to provide a highly aesthetic UX.
- **Real-Time Analytics:** The application renders live SVG confidence rings mapping directly from the Softmax layer to the user's screen in milliseconds.

### ▶️ Quick Start

**1. Clone and Install Dependencies:**
```bash
git clone https://github.com/David-Antony/Huntington-Disease-Detection.git
cd Huntington-Disease-Detection/hf_space
pip install -r requirements.txt

2. Launch the Application:

bash
python app.py
3. Interact: Navigate to http://localhost:7860 locally to interact with the QuantumNeuro dashboard and upload MRI scans.

📁 Repository Structure
text
├── hf_space/                    # Real-time Web App Application 
│   ├── app.py                   # Production inference server
│   ├── Dockerfile               # App Containerization definition
│   ├── requirements.txt         # Production dependencies
│   ├── static/                  # Compiled JS and custom CSS stylesheets
│   └── templates/               # Web Application views (index.html)
├── trainedmodel_file/           # Persisted best model Checkpoints (.pth)
├── train.py                     # Initial base Training Routine
└── train2.py                    # Advanced Training Routine (Full Quantum Stack)
⚠️ Medical Disclaimer
This AI system and the accompanying codebase are provided strictly for research and educational purposes. They are not FDA-approved and should NOT be used as a substitute for professional medical diagnosis. Always consult with a qualified neurologist or healthcare provider regarding Huntington's Disease.
