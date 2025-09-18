import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack implementation.
    
    Based on "Explaining and Harnessing Adversarial Examples" by Goodfellow et al.
    The attack generates adversarial examples by taking a step in the direction
    of the sign of the gradient with respect to the input.
    
    Formula: x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    where:
    - x is the original input
    - epsilon is the perturbation magnitude
    - J is the loss function
    - θ are the model parameters
    - y is the true label
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the FGSM attack.
        
        Args:
            model: PyTorch model to attack
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate_adversarial_example(self, image, target, epsilon=0.1):
        """
        Generate adversarial example using FGSM.
        
        Args:
            image: Input image tensor (normalized)
            target: Target label tensor
            epsilon: Perturbation magnitude in pixel space [0-1] (default: 0.1)
            
        Returns:
            adversarial_image: Adversarial example
            original_pred: Original prediction
            adversarial_pred: Adversarial prediction
            attack_success: Boolean indicating if attack was successful
        """
        # MNIST normalization constants
        MNIST_MEAN = 0.1307
        MNIST_STD = 0.3081
        
        # Convert epsilon from pixel space to normalized space
        epsilon_normalized = epsilon / MNIST_STD
        
        # Ensure image requires gradient
        image = Variable(image.to(self.device), requires_grad=True)
        target = target.to(self.device)
        
        # Forward pass to get original prediction
        output = self.model(image)
        original_pred = output.max(1, keepdim=True)[1].item()
        
        # Calculate loss
        loss = F.cross_entropy(output, target)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Collect gradients of the input
        if image.grad is not None:
            data_grad = image.grad.data
        else:
            raise RuntimeError("No gradients found. Make sure requires_grad=True")
        
        # Generate adversarial example using FGSM
        # x_adv = x + epsilon * sign(gradient)
        perturbed_image = image + epsilon_normalized * data_grad.sign()
        
        # Clip to valid normalized range for MNIST
        # For normalized inputs: min = (0 - mean) / std, max = (1 - mean) / std
        min_val = (0 - MNIST_MEAN) / MNIST_STD
        max_val = (1 - MNIST_MEAN) / MNIST_STD
        perturbed_image = torch.clamp(perturbed_image, min_val, max_val)
        
        # Get adversarial prediction
        with torch.no_grad():
            adv_output = self.model(perturbed_image)
            adversarial_pred = adv_output.max(1, keepdim=True)[1].item()
        
        # Check if attack was successful (prediction changed)
        attack_success = original_pred != adversarial_pred
        
        return perturbed_image, original_pred, adversarial_pred, attack_success
    
    def evaluate_robustness(self, test_loader, epsilon=0.1, max_samples=1000):
        """
        Evaluate model robustness against FGSM attack on a test dataset.
        
        Args:
            test_loader: PyTorch DataLoader with test data
            epsilon: Perturbation magnitude
            max_samples: Maximum number of samples to test
            
        Returns:
            dict: Results containing accuracy metrics
        """
        correct_clean = 0
        correct_adversarial = 0
        successful_attacks = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            if total_samples >= max_samples:
                break
                
            batch_size = data.size(0)
            
            for i in range(batch_size):
                if total_samples >= max_samples:
                    break
                    
                # Get single sample
                single_data = data[i:i+1]
                single_target = target[i:i+1]
                
                # Generate adversarial example
                adv_image, orig_pred, adv_pred, attack_success = self.generate_adversarial_example(
                    single_data, single_target, epsilon
                )
                
                # Check clean accuracy
                if orig_pred == single_target.item():
                    correct_clean += 1
                
                # Check adversarial accuracy
                if adv_pred == single_target.item():
                    correct_adversarial += 1
                
                # Count successful attacks
                if attack_success:
                    successful_attacks += 1
                
                total_samples += 1
        
        clean_accuracy = correct_clean / total_samples
        adversarial_accuracy = correct_adversarial / total_samples
        attack_success_rate = successful_attacks / total_samples
        
        return {
            'total_samples': total_samples,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'attack_success_rate': attack_success_rate,
            'accuracy_drop': clean_accuracy - adversarial_accuracy,
            'epsilon': epsilon
        }


def preprocess_image_for_mnist(image_path):
    """
    Preprocess an uploaded image for MNIST model inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load and convert image to grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize to 28x28 (MNIST input size)
    image = image.resize((28, 28))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    image_tensor = transform(image)  # This returns a tensor
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def tensor_to_base64_image(tensor, unnormalize=True):
    """
    Convert a tensor to base64 encoded image string.
    
    Args:
        tensor: Image tensor
        unnormalize: Whether to unnormalize the tensor (for MNIST)
        
    Returns:
        str: Base64 encoded image
    """
    import base64
    from io import BytesIO
    
    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    
    # Unnormalize if needed (reverse MNIST normalization)
    if unnormalize:
        tensor = tensor * 0.3081 + 0.1307
    
    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    if tensor.shape[0] == 1:  # Grayscale
        tensor = tensor.squeeze(0)
        image = Image.fromarray((tensor.cpu().detach().numpy() * 255).astype(np.uint8), mode='L')
    else:  # RGB
        tensor = tensor.permute(1, 2, 0)
        image = Image.fromarray((tensor.cpu().detach().numpy() * 255).astype(np.uint8))
    
    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.getvalue()).decode()
    
    return base64_string