import torch
import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt
from model import FullArchitecture

def verify_layer_extraction(model, sample_input, device='cpu'):
    """
    Verify that we're correctly extracting features from each layer
    by visualizing a sample and printing intermediate shapes.
    """
    model.eval()
    sample_input = sample_input.to(device)
    
    # Layer 0: Original input
    print(f"Layer 0 (Input) shape: {sample_input.shape}")
    flattened_input = sample_input.view(sample_input.shape[0], -1)
    print(f"Flattened input shape: {flattened_input.shape}")
    
    # Layer 1: After feature extraction
    with torch.no_grad():
        features = model.feature_extractor(sample_input)
    print(f"Layer 1 (After feature extraction) shape: {features.shape}")
    
    # Check if feature extraction actually transforms the data
    if torch.allclose(features, flattened_input):
        print("WARNING: Feature extraction does not transform the data!")
    else:
        print("Feature extraction transforms the data.")
    
    # Layer 2: After denoising
    with torch.set_grad_enabled(True):
        sample_features = features.clone().requires_grad_(True)
        denoised_features, trajectory = model.denoising_potential(sample_features)
    print(f"Layer 2 (After denoising) shape: {denoised_features.shape}")
    
    # Check if denoising transforms the features
    if torch.allclose(features, denoised_features):
        print("WARNING: Denoising does not transform the features!")
    else:
        print("Denoising transforms the features.")
        print(f"L2 distance before vs. after denoising: {torch.norm(features - denoised_features)}")
    
    # Layer 3: After classification
    with torch.no_grad():
        logits = model.classifier(denoised_features)
    print(f"Layer 3 (After classification) shape: {logits.shape}")
    
    return [flattened_input, features, denoised_features, logits]

def verify_fuzziness_calculation(features, labels):
    """
    Verify our implementation of separation fuzziness by
    comparing with alternative formulations and printing intermediates.
    """
    # Ensure on CPU for numpy compatibility
    features = features.cpu()
    labels = labels.cpu()
    
    unique_labels = torch.unique(labels)
    n_classes = len(unique_labels)
    n_samples = features.shape[0]
    feature_dim = features.shape[1]
    
    print(f"Computing separation fuzziness for {n_samples} samples with {n_classes} classes in {feature_dim} dimensions")
    
    # Global mean
    global_mean = torch.mean(features, dim=0)
    
    # Initialize matrices
    ssb = torch.zeros((feature_dim, feature_dim))
    ssw = torch.zeros((feature_dim, feature_dim))
    
    # For class separation visualization
    class_means = []
    
    # Compute between-class and within-class matrices
    for c in unique_labels:
        # Get samples of this class
        class_mask = (labels == c)
        class_features = features[class_mask]
        n_class_samples = class_features.shape[0]
        
        # Class mean
        class_mean = torch.mean(class_features, dim=0)
        class_means.append(class_mean.numpy())
        
        # Between-class contribution
        mean_diff = (class_mean - global_mean).unsqueeze(0)
        ssb += (n_class_samples / n_samples) * torch.mm(mean_diff.T, mean_diff)
        
        # Within-class contribution
        for i in range(n_class_samples):
            sample = class_features[i].unsqueeze(0)
            diff = sample - class_mean.unsqueeze(0)
            ssw += (1 / n_samples) * torch.mm(diff.T, diff)
    
    # Print stats about matrices
    print(f"SSb shape: {ssb.shape}, trace: {torch.trace(ssb).item()}")
    print(f"SSw shape: {ssw.shape}, trace: {torch.trace(ssw).item()}")
    
    # Convert to numpy for pinv calculation
    ssb_np = ssb.numpy()
    ssw_np = ssw.numpy()
    
    # Calculate Moore-Penrose pseudo-inverse of SSb
    ssb_pinv = pinv(ssb_np)
    
    # Calculate separation fuzziness D = Tr(SSw * SS+b)
    product = np.matmul(ssw_np, ssb_pinv)
    D = np.trace(product)
    
    print(f"Separation fuzziness (D): {D}")
    
    # Alternative metrics for comparison
    
    # 1. Average pairwise distance between class means
    class_means = np.array(class_means)
    avg_dist = 0
    count = 0
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            dist = np.linalg.norm(class_means[i] - class_means[j])
            avg_dist += dist
            count += 1
    
    if count > 0:
        avg_dist /= count
        print(f"Average distance between class means: {avg_dist}")
    
    # 2. Ratio of between-class to within-class variance (simple scalar version)
    total_between_var = torch.trace(ssb).item()
    total_within_var = torch.trace(ssw).item()
    if total_within_var > 0:
        ratio = total_between_var / total_within_var
        print(f"Ratio of between-class to within-class variance: {ratio}")
    
    return D, avg_dist, ratio

def visualize_features(features_list, labels, layer_names=None):
    """
    Visualize features at different layers using PCA.
    """
    from sklearn.decomposition import PCA
    
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(len(features_list))]
    
    unique_labels = torch.unique(labels).cpu().numpy()
    n_classes = len(unique_labels)
    
    fig, axes = plt.subplots(1, len(features_list), figsize=(5*len(features_list), 5))
    if len(features_list) == 1:
        axes = [axes]
    
    for i, (features, ax) in enumerate(zip(features_list, axes)):
        # Convert to numpy and apply PCA
        features_np = features.cpu().detach().numpy()
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_np)
        
        # Plot each class with a different color
        for c in unique_labels:
            mask = labels.cpu().numpy() == c
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f"Class {c}", alpha=0.6)
        
        explained_var = pca.explained_variance_ratio_
        ax.set_title(f"{layer_names[i]}\nExplained var: {explained_var[0]:.2f}, {explained_var[1]:.2f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.3)
    
    # Add legend to the last subplot
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig.savefig("layer_features_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return "Visualization saved to layer_features_pca.png"

def run_validation_tests(model_path, dataloader, device='cpu'):
    """
    Run all validation tests on the model.
    """
    try:
        # Load model
        model = FullArchitecture()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        
        # Get a batch of data
        sample_data, sample_labels = next(iter(dataloader))
        
        # Verify layer extraction
        print("\n=== VERIFYING LAYER EXTRACTION ===")
        layer_features = verify_layer_extraction(model, sample_data, device)
        
        # Verify fuzziness calculation with alternative metrics
        print("\n=== VERIFYING FUZZINESS CALCULATION ===")
        
        # Collect results for each layer
        fuzziness_results = []
        avg_dist_results = []
        ratio_results = []
        
        for i, features in enumerate(layer_features):
            print(f"\nLayer {i} analysis:")
            try:
                D, avg_dist, ratio = verify_fuzziness_calculation(features, sample_labels)
                fuzziness_results.append(D)
                avg_dist_results.append(avg_dist)
                ratio_results.append(ratio)
            except Exception as e:
                print(f"Error in fuzziness calculation for layer {i}: {str(e)}")
        
        if fuzziness_results:
            # Compare metrics across layers
            print("\n=== SUMMARY OF METRICS ACROSS LAYERS ===")
            print(f"Separation fuzziness (D): {fuzziness_results}")
            print(f"Average distance between class means: {avg_dist_results}")
            print(f"Ratio of between/within variance: {ratio_results}")
            
            # Visualize features at each layer
            print("\n=== VISUALIZING FEATURES AT EACH LAYER ===")
            layer_names = ["Raw Input", "Feature Extraction", "After Denoising", "Classifier Output"]
            vis_result = visualize_features(layer_features, sample_labels, layer_names)
            print(vis_result)
            
            return {
                "fuzziness": fuzziness_results,
                "avg_distance": avg_dist_results,
                "variance_ratio": ratio_results
            }
        else:
            print("No fuzziness results collected.")
            return None
    
    except Exception as e:
        print(f"Error in validation tests: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_gradient_ascent_trajectory(model, sample_data, device='cpu'):
    """
    Analyze how separation fuzziness changes during gradient ascent.
    """
    model.eval()
    sample_data = sample_data.to(device)
    sample_labels = torch.zeros(sample_data.shape[0], dtype=torch.long, device=device)  # Dummy labels for visualization
    
    try:
        # Get features
        with torch.no_grad():
            features = model.feature_extractor(sample_data)
        
        # Get trajectory from denoising potential
        with torch.set_grad_enabled(True):
            features_var = features.clone().requires_grad_(True)
            _, trajectory = model.denoising_potential(features_var)
        
        # Convert trajectory to numpy for easier handling
        trajectory_np = [t.detach().cpu().numpy() for t in trajectory]
        
        # Visualize trajectory in 2D using PCA
        from sklearn.decomposition import PCA
        
        # Flatten all points from all steps for PCA fitting
        all_points = np.vstack(trajectory_np)
        pca = PCA(n_components=2)
        pca.fit(all_points)
        
        # Transform each step in the trajectory
        trajectory_2d = [pca.transform(step) for step in trajectory_np]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Choose some sample points to track
        n_samples = min(5, trajectory_2d[0].shape[0])
        
        for i in range(n_samples):
            # Extract trajectory for sample i
            point_trajectory = np.array([step[i] for step in trajectory_2d])
            
            # Plot with increasing color intensity
            plt.plot(point_trajectory[:, 0], point_trajectory[:, 1], 'o-', 
                    alpha=0.7, label=f"Sample {i}")
            
            # Mark start and end
            plt.scatter(point_trajectory[0, 0], point_trajectory[0, 1], 
                       color='blue', edgecolor='black', s=100, marker='o')
            plt.scatter(point_trajectory[-1, 0], point_trajectory[-1, 1], 
                       color='red', edgecolor='black', s=100, marker='x')
        
        plt.title("Feature Trajectories During Gradient Ascent")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("gradient_ascent_trajectory.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Trajectory visualization saved to gradient_ascent_trajectory.png")
        
        # Calculate separation fuzziness at each step
        fuzziness_values = []
        
        for step_features in trajectory:
            try:
                step_features_np = step_features.detach().cpu()
                D, _, _ = verify_fuzziness_calculation(step_features_np, sample_labels.cpu())
                fuzziness_values.append(D)
            except Exception as e:
                print(f"Error calculating fuzziness for step: {str(e)}")
        
        # Plot fuzziness vs step
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(fuzziness_values)), fuzziness_values, 'o-', linewidth=2)
        plt.title("Separation Fuzziness During Gradient Ascent")
        plt.xlabel("Gradient Ascent Step")
        plt.ylabel("Separation Fuzziness (D)")
        plt.grid(True, alpha=0.3)
        plt.savefig("fuzziness_trajectory.png", dpi=150)
        plt.close()
        
        print("Fuzziness trajectory saved to fuzziness_trajectory.png")
        
        return fuzziness_values
    
    except Exception as e:
        print(f"Error analyzing gradient ascent: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # This code can be run independently to verify the model
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Load a small subset of test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Run validation tests
        model_path = 'denoising_model.pth'
        results = run_validation_tests(model_path, test_loader, device)
        
        if results:
            # Plot results
            metrics = ['fuzziness', 'avg_distance', 'variance_ratio']
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))
            
            for i, metric in enumerate(metrics):
                axes[i].plot(results[metric], '-o', linewidth=2, markersize=8)
                axes[i].set_title(f"{metric} across layers")
                axes[i].set_xlabel("Layer")
                axes[i].set_ylabel(metric)
                axes[i].grid(alpha=0.3)
                # Add layer labels
                layer_names = ["Input", "Feature Extraction", "Denoising", "Classification"]
                axes[i].set_xticks(range(len(layer_names)))
                axes[i].set_xticklabels(layer_names, rotation=45)
            
            plt.tight_layout()
            fig.savefig("validation_metrics.png", dpi=150)
            print("Validation metrics plot saved to validation_metrics.png")
        
        # Sample data for trajectory analysis
        sample_data, _ = next(iter(test_loader))
        
        # Analyze gradient ascent trajectory
        print("\n=== ANALYZING GRADIENT ASCENT TRAJECTORY ===")
        model = FullArchitecture()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        
        fuzziness_trajectory = analyze_gradient_ascent_trajectory(model, sample_data, device)
        
        if fuzziness_trajectory:
            # Calculate correlation and fit line
            steps = np.arange(len(fuzziness_trajectory))
            log_fuzziness = np.log(fuzziness_trajectory)
            correlation = np.corrcoef(steps, log_fuzziness)[0, 1]
            
            # Fit line to log-scale data
            slope, intercept = np.polyfit(steps, log_fuzziness, 1)
            rho = np.exp(slope)
            
            print(f"\nGradient ascent analysis:")
            print(f"Correlation between step and log(fuzziness): {correlation:.4f}")
            print(f"Estimated rho (slope in log space): {rho:.4f}")
            print(f"Each step {'increases' if rho > 1 else 'decreases'} fuzziness by a factor of {rho:.4f}")
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()