import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np  # Add this too


# to import from model.py
from model import FullArchitecture, train_model, analyze_curl, visualize_trajectories

from visualizations import (
    visualize_centers,
    visualize_trajectory,
    compare_centers_evolution,
    visualize_covariance_structure
)


def load_mnist(batch_size=64):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    feature_dim = 784
    k = 20  # Number of centers (should be ≥ number of classes)
    n_iterations = 30
    epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist(batch_size)

    # Create model
    print("Creating model...")
    model = FullArchitecture(
        feature_dim=feature_dim,
        k=k,
        n_iterations=n_iterations
    ).to(device)

    print("Saving initial centers")
    initial_centers = model.denoising_potential.mu.detach().clone().cpu()


    # Train model
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        device=device,
        lr=learning_rate
    )
        # Generate visualizations
    print("Generating visualizations...")

    # 1. Visualize centers and their weights
    print("Generating visualization for learned centers and distribution of weights")
    visualize_centers(
        model,
        save_path='visualizations/learned_centers.png'
    )

    # 2. Compare initial and final centers
    print("Generating visualization for comparison for initial and final centers")
    compare_centers_evolution(
        initial_centers,
        model.denoising_potential.mu.detach(),
        save_path='visualizations/centers_evolution.png'
    )

  # 3. Visualize trajectory for a few test examples
    print("Visualizing trajectories")
    test_iter = iter(test_loader)
    test_images, _ = next(test_iter)

    # Move images to device and ensure they require gradients
    for i in range(min(5, len(test_images))):
        with torch.set_grad_enabled(True):  # Explicitly enable gradient computation
            img = test_images[i:i+1].clone().to(device)
            img.requires_grad_(True)  # Enable gradients for the image

            try:
                visualize_trajectory(
                    model,
                    img,
                    save_path=f'visualizations/trajectory_example_{i}.png'
                )
            except Exception as e:
                print(f"Error visualizing trajectory for image {i}: {str(e)}")
                continue

    # 4. Analyze covariance structure
    print("Visualizing Covarience Structure")
    visualize_covariance_structure(
        model,
        save_path='visualizations/covariance_structure.png'
    )

    # Optional: Visualize specific center covariances
    for i in range(3):  # Show first 3 centers
        visualize_covariance_structure(
            model,
            center_idx=i,
            save_path=f'visualizations/covariance_center_{i}.png'
        )

    # Save model
    torch.save(model.state_dict(), 'denoising_model.pth')
    print("Model saved as 'denoising_model.pth'")

if __name__ == "__main__":
    main()