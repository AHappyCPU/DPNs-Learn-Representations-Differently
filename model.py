import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class DenoisingPotential(nn.Module):
    def __init__(self, feature_dim, k, n_iterations=10):
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim
        self.n_iterations = n_iterations
        
        # Reparametrized weights: wᵢ = exp(cᵢ)
        self.c = nn.Parameter(torch.zeros(k))
        
        # Centers (μᵢ)
        self.mu = nn.Parameter(torch.randn(k, feature_dim) * 0.1)
        
        # Matrices Aᵢ for Σᵢ⁻¹ = AᵢᵀAᵢ
        self.A = nn.Parameter(torch.eye(feature_dim).unsqueeze(0).repeat(k, 1, 1))
        
        # Gradient ascent step size
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def compute_weights(self):
        """Compute wᵢ from cᵢ"""
        return torch.exp(self.c)

    def compute_precision_matrices(self):
        """Compute Σᵢ⁻¹ = AᵢᵀAᵢ"""
        return torch.bmm(self.A.transpose(1, 2), self.A)

    def compute_potential(self, x):
        """
        Compute φ(x) = log Σᵢ wᵢ exp(-1/2 (x-μᵢ)ᵀΣᵢ⁻¹(x-μᵢ))
        """
        batch_size = x.shape[0]
        weights = self.compute_weights().unsqueeze(0)  # (1,k)
        precision_matrices = self.compute_precision_matrices()  # (k, d, d)
        
        x_expanded = x.unsqueeze(1)  # Shape: (batch_size, 1, d)
        mu_expanded = self.mu.unsqueeze(0)  # Shape: (1, k, d)
        diff = x_expanded - mu_expanded  # Shape: (batch_size, k, d)

        # Compute quadratic terms using einsum
        quad_terms = torch.einsum('bki,kij,bkj->bk', diff, precision_matrices, diff)

        # Compute sum of weighted exponentials using the log-sum-exp trick
        max_quad = torch.max(-0.5 * quad_terms, dim=1, keepdim=True)[0]
        exp_terms = weights * torch.exp(-0.5 * quad_terms - max_quad)
        return torch.log(torch.sum(exp_terms, dim=1)) + max_quad.squeeze(1)


    def compute_gradient(self, x):
        """Compute ∇φ(x)"""
        x = x.detach().requires_grad_(True)
        potential = self.compute_potential(x)
        gradient = grad(potential.sum(), x, create_graph=True)[0]
        return gradient

    def forward(self, x):
        """Apply n iterations of gradient ascent"""
        x_current = x.clone().detach()
        trajectory = [x_current.clone()]
        
        for _ in range(self.n_iterations):
            gradient = self.compute_gradient(x_current)
            x_current = x_current + self.alpha * gradient
            x_current = x_current.detach()
            trajectory.append(x_current.clone())
        
        # Return final point and trajectory for analysis
        return x_current, trajectory

class FullArchitecture(nn.Module):
    def __init__(self, feature_dim=784, k=20, n_iterations=10):
        super().__init__()
        
        # Phase 1: Feature extraction (example for MNIST)
        self.feature_extractor = nn.Flatten()
        
        # Phase 2: Denoising potential
        self.denoising_potential = DenoisingPotential(feature_dim, k, n_iterations)
        
        # Phase 3: Classification
        self.classifier = nn.Linear(feature_dim, 10)  # 10 classes for MNIST

    def forward(self, x, return_trajectory=False):
        # Phase 1: Extract features
        features = self.feature_extractor(x)
        
        # Phase 2: Apply denoising potential
        denoised_features, trajectory = self.denoising_potential(features)
        
        # Phase 3: Classify
        logits = self.classifier(denoised_features)
        
        if return_trajectory:
            return logits, trajectory
        return logits

    def initialize_centers(self, train_loader, device):
        """Initialize μᵢ using k-means on extracted features"""
        self.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                features = self.feature_extractor(x)
                features_list.append(features.cpu())
                labels_list.append(y)
        
        features = torch.cat(features_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()
        
        # Use k-means to initialize centers
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.denoising_potential.k)
        kmeans.fit(features)
        
        # Set centers to k-means centroids
        self.denoising_potential.mu.data = torch.tensor(
            kmeans.cluster_centers_,
            device=device,
            dtype=torch.float32
        )

def train_model(model, train_loader, val_loader, device, epochs, lr=1e-3):
    """Training with comprehensive monitoring"""
    print(f"Starting training for {epochs} epochs")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Initialize centers using training data
    model.initialize_centers(train_loader, device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass with trajectory
            logits, trajectory = model(data, return_trajectory=True)
            loss = criterion(logits, target)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
                
        # Calculate average training metrics for this epoch
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            val_loss += criterion(logits, target).item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch: {epoch}, '
            f'Train Loss: {train_loss / len(train_loader):.4f}, '
            f'Val Loss: {val_loss / len(val_loader):.4f}, '
            f'Val Acc: {100. * correct / total:.2f}%')
        
    print(f"\nMetrics collected:")
    print(f"Train losses: {train_losses}")
    print(f"Val losses: {val_losses}")
    print(f"Train accuracies: {train_accuracies}")
    print(f"Val accuracies: {val_accuracies}")
   
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    print("Saving training curves plot...")
    plt.savefig('training_curves.png')
    plt.close()
    
    # Return the collected metrics
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


# Analysis tools for research questions
def analyze_curl(model, data_loader, device):
    """Analyze curl of the vector field implemented by pretrained resnet blocks"""
    model.eval()
    curls = []
    
    
    for data, _ in data_loader:
        data = data.to(device)
        features = model.feature_extractor(data)
            
            # Compute curl at each point
        for x in features:
            x = x.unsqueeze(0)
            x.requires_grad_(True)
                
                # Compute gradient field
            gradient = model.denoising_potential.compute_gradient(x)
                
            # Approximate curl magnitude using finite differences
            # This is a simplified 2D curl calculation
            h = 1e-5
            curl_z = 0
                
            for i in range(x.shape[1] - 1):
                for j in range(i + 1, x.shape[1]):
                    # Compute partial derivatives
                    x_plus_i = x.clone()
                    x_plus_i[0, i] += h
                    grad_plus_i = model.denoising_potential.compute_gradient(x_plus_i)
                        
                    x_plus_j = x.clone()
                    x_plus_j[0, j] += h
                    grad_plus_j = model.denoising_potential.compute_gradient(x_plus_j)
                        
                    # Approximate curl component
                    curl_component = (
                        (grad_plus_i[0, j] - gradient[0, j]) / h -
                        (grad_plus_j[0, i] - gradient[0, i]) / h
                    )
                    curl_z += curl_component.item() ** 2
                
                curls.append(np.sqrt(curl_z))
    
    return np.array(curls)

def visualize_trajectories(model, data_loader, device):
    """Visualize feature trajectories during gradient ascent"""
    model.eval()
    
    # Get trajectories for a batch
    data, labels = next(iter(data_loader))
    data = data.to(device)
    
    with torch.no_grad():
        _, trajectories = model(data, return_trajectory=True)
    
    # Convert trajectories to numpy for visualization
    trajectories = [t.cpu().numpy() for t in trajectories]
    
    # Use PCA to visualize in 2D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    
    # Flatten all trajectories for PCA fitting
    all_points = np.vstack(trajectories)
    pca.fit(all_points)
    
    # Transform each trajectory
    trajectories_2d = [pca.transform(t) for t in trajectories]
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    
    for i, traj in enumerate(trajectories_2d):
        plt.plot(traj[:, 0], traj[:, 1], '-o', label=f'Sample {i}', alpha=0.5)
    
    plt.title('Feature Trajectories During Gradient Ascent')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()