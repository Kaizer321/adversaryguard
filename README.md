# AdversaryGuard: FGSM Adversarial Attack Demo

📋 **Table of Contents**
- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run Locally](#how-to-run-locally)
- [Deployed URLs](#deployed-urls)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Evaluation Results](#evaluation-results)
- [Examples](#examples)
- [Observations](#observations)
- [Contributing](#contributing)

---

## Overview
AdversaryGuard is a web-based demo for generating and visualizing adversarial examples on MNIST digit classification using the Fast Gradient Sign Method (FGSM). It features a FastAPI backend for model inference and attack generation, and a Next.js frontend for user interaction and visualization.

## Mathematical Background
FGSM is a method for creating adversarial examples—inputs that are intentionally perturbed to fool neural networks. Given an input $x$, label $y$, model parameters $\theta$, and loss $J(\theta, x, y)$, FGSM computes:

$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

where $\epsilon$ controls the perturbation strength. The resulting $x_{adv}$ is visually similar to $x$ but can cause the model to misclassify.

## Project Structure
```
AdversaryGuard/
  app_fgsm.py           # FastAPI backend
  fgsm.py               # FGSM attack logic
  train_mnist.py        # Model training script
  mnist_weights.pth     # Pretrained model weights
  data/MNIST/raw/       # MNIST dataset files
  frontend/             # Next.js frontend
    src/app/            # Frontend app code
    public/             # Static assets
    ...
```

## Installation
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd AdversaryGuard/AdversaryGuard
   ```
2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt  # or install torch, fastapi, uvicorn, pillow, torchvision
   ```
3. **Install frontend dependencies:**
   ```sh
   cd frontend
   npm install
   ```

## How to Run Locally
1. **Start the backend:**
   ```sh
   cd AdversaryGuard
   python app_fgsm.py
   ```
   The backend runs at [http://localhost:8000](http://localhost:8000)

2. **Start the frontend:**
   ```sh
   cd frontend
   npm run dev
   ```
   The frontend runs at [http://localhost:3000](http://localhost:3000)

## Deployed URLs
- **Frontend:** [Add your deployed frontend URL here]
- **Backend:** [Add your deployed backend URL here]

## Usage
- Open the frontend in your browser.
- Upload a grayscale image of a handwritten digit (preferably 28x28 pixels, PNG/JPEG).
- Adjust the epsilon value to control attack strength.
- View the original and adversarial predictions and images.

## API Endpoints
- `GET /` — API info
- `GET /health` — Health check
- `POST /attack` — Generate adversarial example
  - **Form Data:**
    - `image`: PNG/JPEG file
    - `epsilon`: float (0–1, default 0.1)
- `POST /evaluate` — Evaluate model robustness on synthetic data
  - **Form Data:**
    - `epsilon`: float (0–1, default 0.1)

## Evaluation Results
- The backend provides an `/evaluate` endpoint to test model robustness against FGSM on synthetic MNIST-like data.
- Results include clean accuracy, adversarial accuracy, attack success rate, and accuracy drop.

## Examples
- Uploading a clean digit image typically results in correct classification.
- After applying FGSM, the adversarial image looks almost identical, but the model's prediction may change.
- Increasing epsilon makes the attack stronger, but too large values can make the image visibly distorted.

## Observations
- **How did predictions change?**
  - Many adversarial images caused the model to misclassify, even though the changes were imperceptible to humans.
  - The model's confidence in its (wrong) prediction was often high.
- **Did increasing epsilon make attacks stronger?**
  - Yes. Higher epsilon values increased the likelihood of misclassification.
  - Very high epsilon values can make the image look unnatural.

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

**Credits:**
- MNIST dataset
- FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
