# Fashion MNIST CNN Classifier with Grad-CAM Explainability

A convolutional neural network trained on the Fashion MNIST dataset, extended with **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualise *which pixels drove each prediction* — making the model's decisions interpretable.

---

## What this does

| Stage | Details |
|---|---|
| **Model** | 2-layer CNN (Conv2D → MaxPool → Conv2D → MaxPool → Dense) |
| **Dataset** | Fashion MNIST — 70,000 grayscale images across 10 clothing categories |
| **Training** | 10 epochs, Adam optimiser, sparse categorical cross-entropy |
| **Evaluation** | Accuracy/loss curves, confusion matrix, per-class classification report |
| **Explainability** | Grad-CAM heatmaps overlaid on predictions — per random sample and per class |

---

## Results

- **Test accuracy: ~91%** across 10,000 held-out images
- Grad-CAM shows the model correctly attends to:
  - Collar/shoulder region for shirts and coats
  - Sole and outline for shoes and boots
  - Bag shape and handles for bags

---

## Tech stack

- Python 3.10+
- TensorFlow / Keras
- NumPy, Matplotlib
- scikit-learn

---

## How to run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fashion-mnist-cnn-classifier
cd fashion-mnist-cnn-classifier

# Install dependencies
pip install -r requirements.txt

# Open in Jupyter or Colab
jupyter notebook XAI_PROJECT_with_GradCAM.ipynb
```

Or open directly in Google Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/fashion-mnist-cnn-classifier/blob/main/XAI_PROJECT_with_GradCAM.ipynb)

---

## How Grad-CAM works

1. A sub-model exposes the output of the last convolutional layer alongside the final predictions.
2. For a given input image, we record the gradient of the predicted class score with respect to those conv activations using `tf.GradientTape`.
3. Gradients are global-average-pooled to get per-filter importance weights.
4. A weighted sum of the activation maps produces a spatial heatmap.
5. The heatmap is upsampled to the input image size and overlaid — **hot (red/yellow) regions had the highest influence on the prediction**.

---

## What I'd improve next

- Add SHAP (SHapley Additive exPlanations) for feature-level attribution
- Experiment with deeper architectures (ResNet-style blocks) for higher accuracy
- Add a Streamlit demo so anyone can upload a clothing image and see the Grad-CAM overlay live

---

## Author

Thatikonda Kushal — [kushalthatikonda@gmail.com](mailto:kushalthatikonda@gmail.com)
