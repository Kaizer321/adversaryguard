from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from typing import Optional
import uvicorn

from fgsm import FGSMAttack, preprocess_image_for_mnist, tensor_to_base64_image


# Simple MNIST CNN model definition
class SimpleMNISTCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super(SimpleMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # Return logits for F.cross_entropy


# Initialize FastAPI app
app = FastAPI(
    title="FGSM Adversarial Attack API",
    description="Fast Gradient Sign Method (FGSM) adversarial attack demonstration API",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],  # Frontend origins
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables for model and attack
model = None
fgsm_attack = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST class labels
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def initialize_model():
    """Initialize the MNIST model and FGSM attack."""
    global model, fgsm_attack
    
    # Create and load a simple MNIST model
    model = SimpleMNISTCNN()
    
    # Try to load pre-trained weights
    try:
        model.load_state_dict(torch.load('mnist_weights.pth', map_location=device))
        print("Loaded pre-trained MNIST weights")
    except FileNotFoundError:
        print("No pre-trained weights found. Training a quick model...")
        from train_mnist import train_mnist_model
        model = train_mnist_model(epochs=2)  # Quick 2-epoch training
        print("Model training completed")
    
    model.to(device)
    model.eval()
    
    # Initialize FGSM attack
    fgsm_attack = FGSMAttack(model, str(device))
    
    print(f"Model initialized on device: {device}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    try:
        initialize_model()
        print("Model initialization completed successfully")
    except Exception as e:
        print(f"Error during model initialization: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FGSM Adversarial Attack API",
        "endpoints": {
            "/attack": "POST - Generate adversarial examples using FGSM",
            "/health": "GET - Health check"
        },
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/attack")
async def fgsm_attack_endpoint(
    image: UploadFile = File(...),
    epsilon: Optional[float] = Form(0.1)
):
    """
    Generate adversarial example using FGSM attack.
    
    Args:
        image: Uploaded image file (PNG/JPEG)
        epsilon: Perturbation magnitude (default: 0.1)
        
    Returns:
        JSON response with clean prediction, adversarial prediction, 
        base64 adversarial image, and attack success status
    """
    try:
        # Validate epsilon parameter
        if epsilon is None:
            epsilon = 0.1
        if epsilon < 0 or epsilon > 1:
            raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")
        
        # Validate file type
        if image.content_type is None or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await image.read()
        image_stream = io.BytesIO(image_bytes)
        
        # Convert image to PIL and then to tensor
        pil_image = Image.open(image_stream).convert('L')  # Convert to grayscale
        pil_image = pil_image.resize((28, 28))  # Resize to MNIST size
        
        # Normalize for MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        image_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(image_tensor.to(device))
            original_pred_idx = original_output.max(1, keepdim=True)[1].item()
            original_confidence = torch.softmax(original_output, dim=1).max().item()
        
        # Create target tensor (using original prediction as target for untargeted attack)
        target_tensor = torch.tensor([original_pred_idx])
        
        # Generate adversarial example
        adversarial_image, orig_pred, adv_pred, attack_success = fgsm_attack.generate_adversarial_example(
            image_tensor, target_tensor, epsilon
        )
        
        # Get adversarial confidence
        with torch.no_grad():
            adv_output = model(adversarial_image)
            adv_confidence = torch.softmax(adv_output, dim=1).max().item()
        
        # Convert adversarial image to base64
        adversarial_base64 = tensor_to_base64_image(adversarial_image.cpu(), unnormalize=True)
        
        # Convert original image to base64 for comparison
        original_base64 = tensor_to_base64_image(image_tensor, unnormalize=True)
        
        # Prepare response
        response_data = {
            "clean_prediction": {
                "class": MNIST_CLASSES[orig_pred],
                "confidence": float(original_confidence),
                "class_index": int(orig_pred)
            },
            "adversarial_prediction": {
                "class": MNIST_CLASSES[adv_pred],
                "confidence": float(adv_confidence),
                "class_index": int(adv_pred)
            },
            "attack_parameters": {
                "epsilon": float(epsilon),
                "attack_method": "FGSM"
            },
            "attack_success": bool(attack_success),
            "images": {
                "original_base64": original_base64,
                "adversarial_base64": adversarial_base64
            },
            "metadata": {
                "image_size": "28x28",
                "model_type": "SimpleMNISTCNN",
                "device": str(device)
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/evaluate")
async def evaluate_robustness(epsilon: float = Form(0.1)):
    """
    Evaluate model robustness with synthetic MNIST-like data.
    Note: This uses synthetic data for demonstration purposes.
    """
    try:
        if epsilon < 0 or epsilon > 1:
            raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")
        
        # Generate some synthetic test data for demonstration
        synthetic_data = []
        for i in range(10):  # Test 10 synthetic samples
            # Create random MNIST-like images
            random_tensor = torch.randn(1, 1, 28, 28)
            random_tensor = torch.clamp(random_tensor, 0, 1)
            
            # Normalize
            normalized_tensor = (random_tensor - 0.1307) / 0.3081
            
            # Create random target
            target = torch.tensor([i % 10])
            
            synthetic_data.append((normalized_tensor, target))
        
        # Evaluate robustness
        correct_clean = 0
        correct_adversarial = 0
        successful_attacks = 0
        
        for data_tensor, target in synthetic_data:
            # Generate adversarial example
            adv_image, orig_pred, adv_pred, attack_success = fgsm_attack.generate_adversarial_example(
                data_tensor, target, epsilon
            )
            
            # Count accuracies
            if orig_pred == target.item():
                correct_clean += 1
            
            if adv_pred == target.item():
                correct_adversarial += 1
            
            if attack_success:
                successful_attacks += 1
        
        total_samples = len(synthetic_data)
        clean_accuracy = correct_clean / total_samples
        adversarial_accuracy = correct_adversarial / total_samples
        attack_success_rate = successful_attacks / total_samples
        
        return {
            "evaluation_results": {
                "total_samples": total_samples,
                "clean_accuracy": float(clean_accuracy),
                "adversarial_accuracy": float(adversarial_accuracy),
                "attack_success_rate": float(attack_success_rate),
                "accuracy_drop": float(clean_accuracy - adversarial_accuracy),
                "epsilon": float(epsilon)
            },
            "note": "This evaluation uses synthetic data for demonstration purposes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app_fgsm:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )