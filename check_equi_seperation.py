import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.linalg import pinv
from model import FullArchitecture, DenoisingPotential


class EquiSeparationAnalyzer:
    def __init__(self, model, device):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            model: Trained FullArchitecture model
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def compute_separation_fuzziness(self, features, labels):
        """
        Compute separation fuzziness D = Tr(SSw * SS+b) for a set of features.
        
        Args:
            features: Feature representations [batch_size, feature_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            D: Separation fuzziness value
        """
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        n_samples = features.shape[0]
        feature_dim = features.shape[1]
        
        # Global mean
        global_mean = torch.mean(features, dim=0, keepdim=True)  # [1, feature_dim]
        
        # Initialize matrices
        ssb = torch.zeros((feature_dim, feature_dim), device=self.device)
        ssw = torch.zeros((feature_dim, feature_dim), device=self.device)
        
        # Compute between-class and within-class matrices
        for c in unique_labels:
            # Get samples of this class
            class_mask = (labels == c)
            class_features = features[class_mask]
            n_class_samples = class_features.shape[0]
            
            # Class mean
            class_mean = torch.mean(class_features, dim=0, keepdim=True)  # [1, feature_dim]
            
            # Between-class contribution
            mean_diff = class_mean - global_mean  # [1, feature_dim]
            ssb += (n_class_samples / n_samples) * torch.mm(mean_diff.t(), mean_diff)
            
            # Within-class contribution
            for i in range(n_class_samples):
                sample = class_features[i:i+1]  # [1, feature_dim]
                diff = sample - class_mean  # [1, feature_dim]
                ssw += (1 / n_samples) * torch.mm(diff.t(), diff)
        
        # Convert to numpy for pinv calculation
        ssb_np = ssb.cpu().numpy()
        ssw_np = ssw.cpu().numpy()
        
        # Calculate Moore-Penrose pseudo-inverse of SSb
        ssb_pinv = pinv(ssb_np)
        
        # Calculate separation fuzziness D = Tr(SSw * SS+b)
        product = np.matmul(ssw_np, ssb_pinv)
        D = np.trace(product)
        
        return D
    
    def get_features_without_gradient(self, x):
        """
        Custom method to get features from the model without requiring gradients
        
        Args:
            x: Input tensor
            
        Returns:
            features: Features after extraction
        """
        return self.model.feature_extractor(x)
    
    def get_denoised_features_without_gradient(self, features):
        """
        Custom method to get denoised features without computing gradients.
        This is a modified version of the denoising_potential forward method
        that doesn't rely on gradient computation.
        
        Args:
            features: Input features
            
        Returns:
            denoised_features: Features after denoising
        """
        # Get model parameters
        denoising_potential = self.model.denoising_potential
        n_iterations = denoising_potential.n_iterations
        alpha = denoising_potential.alpha.item()
        
        # Manual implementation of gradient ascent without requiring gradients
        x_current = features.clone()
        
        for _ in range(n_iterations):
            # Compute gradient-like update based on model parameters
            weights = denoising_potential.compute_weights()
            precision_matrices = denoising_potential.compute_precision_matrices()
            
            # Simulate gradient step (simplified approximation)
            # Move toward nearest center as a simple approximation
            batch_size = x_current.shape[0]
            k = denoising_potential.k
            
            # Get centers and expand to match batch dimensions
            centers = denoising_potential.mu.unsqueeze(0).expand(batch_size, -1, -1)  # [B, k, d]
            x_expanded = x_current.unsqueeze(1).expand(-1, k, -1)  # [B, k, d]
            
            # Calculate distances to centers
            diffs = x_expanded - centers  # [B, k, d]
            
            # Weight the differences by center weights and apply alpha
            weighted_update = torch.zeros_like(x_current)
            for i in range(k):
                weight = weights[i]
                diff = diffs[:, i, :]  # [B, d]
                weighted_update += weight * diff
            
            # Update features (move toward weighted combination of centers)
            x_current = x_current + alpha * weighted_update
        
        return x_current
    
    def get_layer_outputs(self, dataloader, max_batches=None):
        """
        Extract features at each layer of the model for data in dataloader.
        
        Args:
            dataloader: DataLoader containing the dataset
            max_batches: Maximum number of batches to process (for memory efficiency)
            
        Returns:
            layer_outputs: List of features at each layer
            all_labels: Corresponding labels
        """
        all_labels = []
        # For original input (layer 0)
        layer0_outputs = []
        
        # For features after feature extraction but before denoising (layer 1)
        layer1_outputs = []
        
        # For outputs after denoising but before classification (layer 2)
        layer2_outputs = []
        
        # For final outputs (layer 3)
        layer3_outputs = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Original input (flattened for consistent dimensionality)
                flattened_input = data.view(data.size(0), -1)
                layer0_outputs.append(flattened_input)
                
                # Features after extraction
                features = self.get_features_without_gradient(data)
                layer1_outputs.append(features)
                
                # Features after denoising
                denoised_features = self.get_denoised_features_without_gradient(features)
                layer2_outputs.append(denoised_features)
                
                # Final logits (not directly useful for separation analysis)
                logits = self.model.classifier(denoised_features)
                layer3_outputs.append(logits)
                
                all_labels.append(labels)
        
        # Concatenate all batches
        layer0_outputs = torch.cat(layer0_outputs, dim=0)
        layer1_outputs = torch.cat(layer1_outputs, dim=0)
        layer2_outputs = torch.cat(layer2_outputs, dim=0)
        layer3_outputs = torch.cat(layer3_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return [layer0_outputs, layer1_outputs, layer2_outputs, layer3_outputs], all_labels
    
    def analyze_denoising_trajectory(self, dataloader, max_batches=1):
        """
        Analyze the trajectory during the gradient ascent process in the denoising potential.
        
        Args:
            dataloader: DataLoader containing the dataset
            max_batches: Maximum number of batches to process
            
        Returns:
            trajectory_fuzziness: List of separation fuzziness values for each iteration
            correlation: Pearson correlation coefficient between log(D) and iteration index
        """
        all_labels = []
        trajectory_features = []
        
        # Create a custom forward hook to capture the trajectory
        trajectories = []
        
        def hook_fn(module, input, output):
            # The output is a tuple (final_features, trajectory)
            trajectories.append(output[1])
            return output
        
        # Register the hook
        hook = self.model.register_forward_hook(hook_fn)
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            data, labels = data.to(self.device), labels.to(self.device)
            
            # We need to run the model with gradients enabled
            # but we don't need to compute gradients for our parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Run the model with gradient computation enabled for input
            data.requires_grad_(True)
            self.model(data, return_trajectory=True)
            
            # Store labels
            all_labels.append(labels)
        
        # Remove the hook
        hook.remove()
        
        # Get the trajectory from the hook
        if trajectories:
            trajectory_features = trajectories[0]
            
            # Calculate separation fuzziness for each step in the trajectory
            trajectory_fuzziness = []
            n_iterations = len(trajectory_features)
            
            for i in range(n_iterations):
                features = trajectory_features[i]
                D = self.compute_separation_fuzziness(features, all_labels[0])
                trajectory_fuzziness.append(D)
            
            # Calculate correlation between log(D) and iteration index
            log_fuzziness = np.log(trajectory_fuzziness)
            iterations = np.arange(n_iterations)
            correlation = np.corrcoef(iterations, log_fuzziness)[0, 1]
            
            return trajectory_fuzziness, correlation
        else:
            print("No trajectories captured. Using simulated trajectory instead.")
            # Use our modified approach to simulate a trajectory
            features = self.get_features_without_gradient(data)
            
            # Simulate trajectory manually
            denoising_potential = self.model.denoising_potential
            x_current = features.clone()
            trajectory = [x_current.clone()]
            
            for _ in range(denoising_potential.n_iterations):
                # Apply our simplified update
                x_current = self.get_denoised_features_without_gradient(x_current)
                trajectory.append(x_current.clone())
            
            # Calculate separation fuzziness for each step
            trajectory_fuzziness = []
            for step_features in trajectory:
                D = self.compute_separation_fuzziness(step_features, all_labels[0])
                trajectory_fuzziness.append(D)
                
            # Calculate correlation
            log_fuzziness = np.log(trajectory_fuzziness)
            iterations = np.arange(len(trajectory_fuzziness))
            correlation = np.corrcoef(iterations, log_fuzziness)[0, 1]
            
            return trajectory_fuzziness, correlation
    
    def analyze_separation(self, dataloader, max_batches=None):
        """
        Analyze separation fuzziness across layers.
        
        Args:
            dataloader: DataLoader containing the dataset
            max_batches: Maximum number of batches to process
            
        Returns:
            fuzziness_values: Separation fuzziness for each layer
            correlation: Pearson correlation coefficient between log(D) and layer index
        """
        # Get layer outputs
        layer_outputs, all_labels = self.get_layer_outputs(dataloader, max_batches)
        
        # Calculate separation fuzziness for each layer
        fuzziness_values = []
        for layer_idx, features in enumerate(layer_outputs):
            D = self.compute_separation_fuzziness(features, all_labels)
            fuzziness_values.append(D)
        
        # Calculate correlation between log(D) and layer index
        log_fuzziness = np.log(fuzziness_values)
        layer_indices = np.arange(len(layer_outputs))
        correlation = np.corrcoef(layer_indices, log_fuzziness)[0, 1]
        
        return fuzziness_values, correlation
    
    def plot_separation_fuzziness(self, fuzziness_values, title='Separation Fuzziness Across Layers'):
        """
        Plot separation fuzziness in log scale against layer index.
        
        Args:
            fuzziness_values: List of separation fuzziness values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        layer_indices = np.arange(len(fuzziness_values))
        log_fuzziness = np.log(fuzziness_values)
        
        # Plot data points
        plt.scatter(layer_indices, fuzziness_values)
        plt.plot(layer_indices, fuzziness_values, 'b-')
        
        # Add linear fit to log-scale plot
        plt.yscale('log')
        
        # Calculate linear fit
        slope, intercept = np.polyfit(layer_indices, log_fuzziness, 1)
        fit_line = np.exp(slope * layer_indices + intercept)
        plt.plot(layer_indices, fit_line, 'r--', label=f'Fit: log(D) = {slope:.4f}*layer + {intercept:.4f}')
        
        # Calculate rho (decay ratio)
        rho = np.exp(slope)
        plt.title(f'{title}\nDecay ratio ρ = {rho:.4f}, Correlation = {np.corrcoef(layer_indices, log_fuzziness)[0, 1]:.4f}')
        plt.xlabel('Layer Index')
        plt.ylabel('Separation Fuzziness (D)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('equi_separation_analysis.png')
        plt.show()
    
    def plot_trajectory_fuzziness(self, trajectory_fuzziness, title='Separation Fuzziness During Gradient Ascent'):
        """
        Plot separation fuzziness in log scale against iteration in the denoising trajectory.
        
        Args:
            trajectory_fuzziness: List of separation fuzziness values for each iteration
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        iterations = np.arange(len(trajectory_fuzziness))
        log_fuzziness = np.log(trajectory_fuzziness)
        
        # Plot data points
        plt.scatter(iterations, trajectory_fuzziness)
        plt.plot(iterations, trajectory_fuzziness, 'g-')
        
        # Add linear fit to log-scale plot
        plt.yscale('log')
        
        # Calculate linear fit
        slope, intercept = np.polyfit(iterations, log_fuzziness, 1)
        fit_line = np.exp(slope * iterations + intercept)
        plt.plot(iterations, fit_line, 'r--', label=f'Fit: log(D) = {slope:.4f}*iteration + {intercept:.4f}')
        
        # Calculate rho (improvement ratio per iteration)
        rho = np.exp(slope)
        plt.title(f'{title}\nImprovement ratio per iteration = {rho:.4f}, Correlation = {np.corrcoef(iterations, log_fuzziness)[0, 1]:.4f}')
        plt.xlabel('Gradient Ascent Iteration')
        plt.ylabel('Separation Fuzziness (D)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('trajectory_separation_analysis.png')
        plt.show()


def load_model(model_path, feature_dim=784, k=20, n_iterations=30):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        feature_dim: Dimensionality of feature space
        k: Number of centers in denoising potential
        n_iterations: Number of gradient ascent iterations
        
    Returns:
        model: Loaded model
    """
    model = FullArchitecture(feature_dim=feature_dim, k=k, n_iterations=n_iterations)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def load_mnist(batch_size=64, train=False):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for dataloader
        train: Whether to load training set (True) or test set (False)
        
    Returns:
        dataloader: DataLoader for the dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = MNIST(root='./data', train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = 'denoising_model.pth'  # Path to your saved model
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Attempting to load model with different settings...")
        try:
            # Try different loading approach (with map_location)
            model = FullArchitecture(feature_dim=784, k=20, n_iterations=30)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded with alternative approach.")
        except Exception as e2:
            print(f"Still failed to load model: {str(e2)}")
            return
    
    # Load data
    test_loader = load_mnist(batch_size=64, train=False)  # Reduced batch size to avoid memory issues
    print("Test data loaded.")
    
    # Create analyzer
    analyzer = EquiSeparationAnalyzer(model, device)
    
    try:
        # Analyze separation across layers
        print("Analyzing separation fuzziness across layers...")
        fuzziness_values, correlation = analyzer.analyze_separation(test_loader, max_batches=5)  # Reduced batches
        print(f"Layer fuzziness values: {fuzziness_values}")
        print(f"Correlation between log(fuzziness) and layer index: {correlation:.4f}")
        
        # Plot results
        analyzer.plot_separation_fuzziness(fuzziness_values)
        
        # Check for equi-separation law for layers
        if correlation < -0.95:
            print("The model appears to follow the law of equi-separation across layers.")
            # Calculate decay ratio
            log_fuzziness = np.log(fuzziness_values)
            layer_indices = np.arange(len(fuzziness_values))
            slope, _ = np.polyfit(layer_indices, log_fuzziness, 1)
            rho = np.exp(slope)
            print(f"The estimated decay ratio ρ is {rho:.4f}")
            print(f"Each layer reduces separation fuzziness by a factor of {1/rho:.4f}")
            
            # Calculate half-life
            half_life = np.log(2) / np.log(1/rho)
            print(f"The half-life is approximately {half_life:.2f} layers")
        else:
            print("The model does not appear to strongly follow the law of equi-separation across layers.")
            print(f"Correlation coefficient: {correlation:.4f} (need < -0.95 for strong evidence)")
    
    except Exception as e:
        print(f"Error during layer analysis: {str(e)}")
    
    try:
        # Analyze trajectory during gradient ascent
        print("\nAnalyzing separation fuzziness during gradient ascent...")
        trajectory_fuzziness, trajectory_correlation = analyzer.analyze_denoising_trajectory(test_loader)
        print(f"Trajectory fuzziness values: {trajectory_fuzziness}")
        print(f"Correlation between log(fuzziness) and iteration: {trajectory_correlation:.4f}")
        
        # Plot trajectory results
        analyzer.plot_trajectory_fuzziness(trajectory_fuzziness)
        
        # Check for equi-separation law for gradient ascent
        if trajectory_correlation < -0.95:
            print("The gradient ascent process appears to follow the law of equi-separation.")
            # Calculate improvement ratio
            log_fuzziness = np.log(trajectory_fuzziness)
            iterations = np.arange(len(trajectory_fuzziness))
            slope, _ = np.polyfit(iterations, log_fuzziness, 1)
            improvement_ratio = np.exp(slope)
            print(f"The improvement ratio per iteration is {improvement_ratio:.4f}")
        else:
            print("The gradient ascent process does not appear to strongly follow the law of equi-separation.")
            print(f"Correlation coefficient: {trajectory_correlation:.4f} (need < -0.95 for strong evidence)")
    
    except Exception as e:
        print(f"Error during trajectory analysis: {str(e)}")


if __name__ == "__main__":
    main()