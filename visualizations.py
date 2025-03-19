import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_centers(model, save_path):
    """Visualize the learned centers and distribution of weights."""
    with torch.no_grad():
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        centers = model.denoising_potential.mu.detach().cpu()
        weights = model.denoising_potential.compute_weights().detach().cpu()

        k = centers.shape[0]
        feature_dim = centers.shape[1]

        img_size = int(np.sqrt(feature_dim))
        if img_size * img_size != feature_dim:
            print(f"Cannot reshape feature_dim {feature_dim} into square images.")
            return

        # Sort centers by weights
        sorted_indices = torch.argsort(weights, descending=True)
        centers = centers[sorted_indices]
        weights = weights[sorted_indices]

        # Normalize centers for visualization
        centers = (centers - centers.min()) / (centers.max() - centers.min())

        n_centers_to_display = min(16, k)

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(n_centers_to_display):
            center_img = centers[i].reshape(img_size, img_size).numpy()
            axes[i].imshow(center_img, cmap='gray')
            axes[i].set_title(f"w={weights[i]:.2f}")
            axes[i].axis('off')
        
        for i in range(n_centers_to_display, 16):
            axes[i].axis('off')

        plt.suptitle("Learned Centers (sorted by weight)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Plot distribution of weights
        plt.figure()
        plt.bar(range(k), weights.numpy())
        plt.xlabel('Center index')
        plt.ylabel('Weight')
        plt.title('Distribution of Weights')
        weights_save_path = save_path.replace('.png', '_weights.png')
        plt.savefig(weights_save_path)
        plt.close()

def compare_centers_evolution(initial_centers, final_centers, save_path):
    """Compare initial and final centers."""
    with torch.no_grad():
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        initial_centers = initial_centers.cpu()
        final_centers = final_centers.cpu()

        k = initial_centers.shape[0]
        feature_dim = initial_centers.shape[1]

        img_size = int(np.sqrt(feature_dim))
        if img_size * img_size != feature_dim:
            print(f"Cannot reshape feature_dim {feature_dim} into square images.")
            return

        n_centers_to_display = min(8, k)

        fig, axes = plt.subplots(2, n_centers_to_display, figsize=(2 * n_centers_to_display, 4))
        for i in range(n_centers_to_display):
            init_center_img = initial_centers[i].reshape(img_size, img_size).numpy()
            final_center_img = final_centers[i].reshape(img_size, img_size).numpy()
            axes[0, i].imshow(init_center_img, cmap='gray')
            axes[0, i].set_title(f"Center {i} (Initial)")
            axes[0, i].axis('off')
            axes[1, i].imshow(final_center_img, cmap='gray')
            axes[1, i].set_title(f"Center {i} (Final)")
            axes[1, i].axis('off')

        plt.suptitle("Comparison of Initial and Final Centers")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def visualize_trajectory(model, input_image, save_path):
    """
    Visualize the trajectory of the gradient ascent starting from the input image.
    
    Args:
        model: The trained model
        input_image: Input image tensor of shape (1, C, H, W)
        save_path: Path to save the visualization
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Ensure input_image requires gradients
    if not input_image.requires_grad:
        input_image = input_image.detach().requires_grad_(True)

    # Get device
    device = next(model.parameters()).device
    input_image = input_image.to(device)

    # Forward pass with trajectory
    with torch.enable_grad():
        logits, trajectory = model(input_image, return_trajectory=True)

    # Process trajectory
    feature_dim = trajectory[0].shape[1]
    img_size = int(np.sqrt(feature_dim))
    
    if img_size * img_size != feature_dim:
        print(f"Cannot reshape feature_dim {feature_dim} into square images.")
        return

    # Plot trajectory
    n_steps = len(trajectory)
    fig, axes = plt.subplots(1, n_steps, figsize=(2 * n_steps, 2))
    
    # Handle case where n_steps = 1
    if n_steps == 1:
        axes = [axes]
    
    for i, features in enumerate(trajectory):
        # Detach and move to CPU for visualization
        features = features.detach().cpu()
        img = features[0].reshape(img_size, img_size).numpy()
        
        # Normalize for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Step {i}")
        axes[i].axis('off')

    plt.suptitle("Feature Trajectory During Gradient Ascent")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_covariance_structure(model, save_path, center_idx=None):
    """Visualize the covariance structure of the model."""
    with torch.no_grad():
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        A = model.denoising_potential.A.detach().cpu()  # Shape: (k, d, d)
        k, d, _ = A.shape

        img_size = int(np.sqrt(d))
        if img_size * img_size != d:
            print(f"Cannot reshape feature_dim {d} into square images.")
            return

        if center_idx is not None:
            if center_idx >= k:
                print(f"Center index {center_idx} is out of range.")
                return
                
            A_i = A[center_idx]  # Shape: (d, d)
            precision_matrix = A_i.t() @ A_i  # Shape: (d, d)
            
            # Add small constant to diagonal for numerical stability
            epsilon = 1e-6
            precision_matrix.diagonal().add_(epsilon)
            
            # Compute covariance matrix
            covariance_matrix = torch.inverse(precision_matrix)
            covariance_matrix = covariance_matrix.numpy()
            
            # Extract variances
            variances = np.diag(covariance_matrix)
            variance_img = variances.reshape(img_size, img_size)
            
            plt.figure()
            plt.imshow(variance_img, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Variance Map for Center {center_idx}')
            plt.savefig(save_path)
            plt.close()
        else:
            # Average covariance matrix
            covariance_matrices = []
            for i in range(k):
                A_i = A[i]
                precision_matrix = A_i.t() @ A_i
                
                # Add small constant to diagonal for numerical stability
                epsilon = 1e-6
                precision_matrix.diagonal().add_(epsilon)
                
                covariance_matrix = torch.inverse(precision_matrix)
                covariance_matrices.append(covariance_matrix.numpy())
            
            covariance_matrices = np.array(covariance_matrices)
            avg_covariance = np.mean(covariance_matrices, axis=0)
            variances = np.diag(avg_covariance)
            variance_img = variances.reshape(img_size, img_size)
            
            plt.figure()
            plt.imshow(variance_img, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('Average Variance Map')
            plt.savefig(save_path)
            plt.close()