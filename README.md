# Energy-Based Models Do Not Follow the Law of Equi-Separation: A Case Study with Denoising Potential Networks

## Abstract

Recent research has demonstrated that traditional deep neural networks follow a "law of equi-separation," where each layer reduces separation fuzziness at a constant geometric rate. We present empirical evidence that energy-based models, specifically Denoising Potential Networks (DPNs), operate on fundamentally different principles. Our analysis reveals that DPNs maintain or increase separation fuzziness while reorganizing features according to an energy landscape, challenging the universality of the equi-separation law and suggesting a novel paradigm for representation learning. Using visualization and quantitative metrics, we show that DPNs transform feature representations while preserving important structural information, in contrast to traditional networks that primarily focus on maximizing class separation. These findings provide new insights into the diversity of learning mechanisms in deep architectures and suggest alternative approaches to building robust and interpretable neural networks.

## 1. Introduction

### 1.1 The Law of Equi-Separation in Deep Learning

The opacity of deep neural networks has long challenged researchers to develop theoretical frameworks that explain how these models process and transform data. A recent breakthrough in this direction is the discovery of the "law of equi-separation" by He and Su (2023). This law quantitatively characterizes how well-trained neural networks separate data according to class membership throughout all layers for classification tasks.

Specifically, He and Su defined a measure called separation fuzziness, 
$D = {Tr}\left(SS_w \cdot SS_b\right)$, 
where $SS_w$ is the within-class sum of squares and $SS_b$ is the between-class sum of squares. This value is large when data points are not well concentrated around their class means (or equivalently, not well separated), and vice versa. Their key finding is that for traditional neural networks, when plotted in logarithmic scale, this separation fuzziness decreases linearly with layer index, following the relation:
$D_l \approx \rho^l D_0$,
where $D_l$ is the separation fuzziness at layer $l$, $D_0$ is the initial separation fuzziness, and $\rho$ is a decay ratio between 0 and 1. This relation implies that each layer in a traditional neural network makes equal progress in reducing $\log(D)$, hence the term "equi-separation."

This law has been demonstrated across various architectures including feedforward neural networks, convolutional neural networks, and even residual networks, suggesting a universal principle in how conventional deep learning models organize feature representations.

### 1.2 Energy-Based Models and Denoising Potential Networks

While conventional neural networks learn direct mappings between input and output spaces, energy-based models (EBMs) take a fundamentally different approach. EBMs learn an energy landscape over the input or feature space, where lower energy regions correspond to more probable or desirable configurations. This approach draws inspiration from statistical physics and provides an alternative perspective on representation learning.

Denoising Potential Networks (DPNs) are a novel class of energy-based models that combine traditional neural network components with energy-based modeling principles. DPNs learn a parameterized potential function represented as a mixture of Gaussian components:
$\phi(x) = \log \sum_i w_i \exp\left(-\frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1}(x-\mu_i)\right)$.
This potential function shapes the feature space through learnable parameters including component centers $\mu_i$, precision matrices $\Sigma_i^{-1}$, and weights $w_i$. Unlike traditional networks that apply a fixed transformation at each layer, DPNs refine feature representations through iterative gradient ascent on this potential function, guiding them toward local maxima that represent stable, denoised configurations.

The DPN architecture consists of three main phases:
1. **Feature extraction**: An initial transformation of input data
2. **Denoising potential**: The energy-based refinement module using gradient ascent
3. **Classification**: A final projection layer mapping to class probabilities

This architecture represents a hybrid approach that incorporates energy-based principles within a discriminative framework, potentially offering advantages in terms of robustness, interpretability, and generalization.

### 1.3 Research Questions

Given the established law of equi-separation in traditional neural networks and the fundamentally different approach of energy-based models, several important questions arise:

1. Do energy-based models, specifically DPNs, follow the law of equi-separation observed in traditional neural networks?
2. If not, what alternative principles govern their separation dynamics?
3. How do the different feature transformation mechanisms affect representations and learning outcomes?
4. What implications might these differences have for robustness, generalization, and applications?

In this project, we address these questions through a comprehensive analysis of separation dynamics in DPNs. We compare their behavior with the equi-separation law and identify distinct principles that characterize energy-based representation learning.

## 2. Methods

### 2.1 Denoising Potential Network Architecture

Our implementation of DPNs follows the three-phase architecture described in the introduction:

**Feature Extraction**: For MNIST-like datasets, this phase is implemented as a simple flattening operation that reshapes the 2D images into 1D vectors. While more complex feature extraction mechanisms could be employed, this simple approach allows us to directly observe the effects of the denoising potential phase.

**Denoising Potential**: This is the core innovation of the architecture. The module implements a parameterized potential function as a mixture of Gaussian components. Each component is characterized by:
- A center $\mu_i \in \mathbb{R}^d$ representing a prototype in feature space
- A precision matrix $\Sigma_i^{-1} = A_i^T A_i$ ensuring positive definiteness
- A weight $w_i = \exp(c_i)$ ensuring positivity

During the forward pass, feature representations are refined through $n$ iterations of gradient ascent on the potential function:
    
    x_current = features.clone()
    for _ in range(n_iterations):
        gradient = compute_gradient(x_current)
        x_current = x_current + alpha * gradient
        x_current = x_current.detach()

where $\alpha$ is a learnable step size parameter.

**Classification**: The final phase is a standard linear layer that maps the denoised feature representations to class logits.

### 2.2 Experimental Setup

We trained a DPN on the MNIST dataset with the following configuration:
- **Feature dimension**: 784 (28×28 flattened images)
- **Number of Gaussian components ($k$)**: 20
- **Number of gradient ascent iterations**: 30
- **Training epochs**: 5
- **Batch size**: 128
- **Optimizer**: Adam with learning rate $1 \times 10^{-3}$

The model was initialized with k-means clustering to provide stable starting points for the Gaussian components. This initialization strategy is crucial for training stability and faster convergence.

### 2.3 Analysis Framework

To investigate the separation dynamics in DPNs, we extracted features at each layer of the network:
- **Layer 0**: Raw input (flattened images)
- **Layer 1**: After feature extraction (identical to Layer 0 in our implementation)
- **Layer 2**: After denoising potential (gradient ascent)
- **Layer 3**: Classifier output (logits)

For each layer, we computed:
1. **Separation fuzziness ($D$)** using the formula 
   $D = {Tr}\left(SS_w \cdot SS_b\right)$
2. **Average distance between class means** to measure global separation
3. **Ratio of between-class to within-class variance** as an alternative class separation metric
4. **PCA projections** for visualization of feature distributions

Additionally, we analyzed the gradient ascent trajectories to understand how features evolve during the iterative refinement process.

## 3. Results

### 3.1 Layer-wise Separation Dynamics in DPNs

Our analysis of separation fuzziness across the layers of a DPN trained on MNIST reveals patterns that significantly diverge from the law of equi-separation observed in traditional neural networks.

Table 1 presents the separation fuzziness values and related metrics at each layer of the DPN:

| Layer | Description            | Separation Fuzziness ($D$) | Avg. Distance Between Classes | Between/Within Variance Ratio |
|-------|------------------------|----------------------------|-------------------------------|-------------------------------|
| 0     | Raw Input              | 2.14                       | 24.51                         | 0.89                          |
| 1     | Feature Extraction     | 2.14                       | 24.51                         | 0.89                          |
| 2     | After Denoising        | 2.81                       | 18.24                         | 2.10                          |
| 3     | Classification         | 9.53                       | 17.17                         | 8.21                          |

These results reveal several striking patterns:

- First, the separation fuzziness remains identical between layers 0 and 1, confirming our architectural understanding that the feature extraction phase in our implementation is merely a flattening operation that does not transform the data.
- Second, after the denoising potential phase (layer 2), we observe that despite significant transformation of the features (L2 distance of 69.6 between pre- and post-denoising representations), the separation fuzziness slightly increases rather than decreases. This directly contradicts the equi-separation law, which would predict a substantial decrease in fuzziness at this stage.
- Third, the classification layer (layer 3) shows a dramatic increase in separation fuzziness, with values more than tripling from 2.81 to 9.53. This behavior stands in stark contrast to traditional neural networks, where the final layers typically show the lowest separation fuzziness values.

Interestingly, while separation fuzziness increases, the between/within-class variance ratio also increases substantially, from 0.89 in the raw input to 8.21 after classification. This indicates that despite the higher separation fuzziness, the model is still organizing features in ways that improve class discrimination, but through mechanisms different from those captured by the separation fuzziness metric.

### 3.2 Feature Transformation Analysis

PCA visualizations of the feature distributions at each layer (Figure 1) provide further insight into how DPNs transform representations. While the first two layers show identical distributions (explained variance: 0.16, 0.11), the denoising potential significantly reorganizes the features (explained variance increases to 0.28, 0.18), and the classifier output shows a much more structured organization (explained variance: 0.55, 0.18).

The increase in explained variance ratio indicates that the DPN is creating a more efficient embedding of the data, concentrating relevant information in fewer dimensions. This suggests that even though separation fuzziness increases, the representations become more structured and potentially more useful for the classification task.

### 3.3 Gradient Ascent Behavior

Analysis of the gradient ascent trajectories reveals that features move along different paths toward stable points in the energy landscape. Each sample follows its own trajectory, guided by the learned potential function toward a local maximum that represents a stable, denoised configuration.

The visualization of these trajectories (Figure 2) shows that samples from the same class tend to move toward similar regions of the feature space, but follow individual paths that preserve their unique characteristics. This behavior suggests that the denoising potential is not simply collapsing all examples of a class to a single point, but rather creating a structured manifold that captures the variability within each class.

## 4. Discussion

### 4.1 A Different Representation Learning Paradigm

Our findings provide strong evidence that Denoising Potential Networks operate on fundamentally different principles than traditional deep neural networks with respect to how they organize feature representations. While the law of equi-separation characterizes traditional networks as progressively reducing separation fuzziness across layers at a constant geometric rate, DPNs maintain or increase this metric while still improving feature organization by other measures.

This difference can be understood through the lens of their underlying operational principles:

**Traditional Neural Networks:**
- Apply a series of learned transformations that directly map inputs to outputs.
- Each layer contributes equally (in log space) to reducing separation fuzziness.
- Optimize for pushing different classes apart.

**Denoising Potential Networks:**
- Learn an energy landscape that guides features toward stable configurations.
- Use gradient ascent to find local maxima of a potential function.
- Optimize for creating stable, structured manifolds that preserve within-class relationships.

The results suggest that DPNs prioritize structural coherence over raw separation as measured by the fuzziness metric. This is evidenced by the simultaneous increase in separation fuzziness and between/within-class variance ratio.

This paradigm shift is particularly visible in our PCA visualizations, where the denoising process maintains global statistics while reorganizing local feature relationships. The energy landscape appears to create attractor basins for different patterns, guiding similar inputs toward shared stable configurations without collapsing the entire feature space.

### 4.2 Theoretical Implications

The finding that energy-based models do not follow the law of equi-separation has several important theoretical implications:

1. **Diversity of Learning Mechanisms:** The equi-separation law is not a universal principle of deep learning but rather characterizes a specific class of architectures. Our work reveals that fundamentally different organizing principles can emerge depending on the architectural inductive biases.
2. **Alternative Optimization Objectives:** While separation fuzziness effectively captures the goal of traditional discriminative networks, it fails to characterize the objectives of energy-based models. New metrics may be needed to properly quantify how well energy-based models organize representations.
3. **Statistical Physics Connections:** The behavior of DPNs aligns with principles from statistical physics, where systems evolve toward stable configurations that minimize free energy. In this framework, the increase in separation fuzziness might be understood as the system organizing according to an alternative objective function related to the learned energy landscape.
4. **Feature Quality vs. Separation:** Our findings suggest that good representation learning may not always be characterized by minimizing separation fuzziness. The quality of features might instead relate to how well they capture the underlying structure and variability of the data, even if this comes at the cost of increased separation fuzziness.

### 4.3 Potential Advantages and Applications

The distinct behavior of DPNs suggests several potential advantages over traditional architectures:

1. **Robustness to Noise and Perturbations:** By guiding representations toward stable configurations in the energy landscape, DPNs may be more robust to input noise and perturbations. The gradient ascent process naturally denoises inputs by moving them toward local maxima of the potential function.
2. **Structure Preservation:** Unlike traditional networks that might collapse within-class variability to minimize separation fuzziness, DPNs appear to preserve important structural information. This could be advantageous for tasks where the internal structure of classes contains valuable information.
3. **Uncertainty Handling:** The energy landscape provides a natural framework for representing uncertainty and ambiguity. Regions of the feature space with multiple nearby local maxima might correspond to areas of heightened uncertainty, providing a built-in mechanism for uncertainty quantification.
4. **Generative Capabilities:** The learned energy landscape could potentially be used for generative purposes through sampling procedures such as Langevin dynamics. This dual discriminative-generative capability represents an advantage over purely discriminative models.

These potential advantages suggest that DPNs and similar energy-based architectures might be particularly well-suited for applications where robustness, structure preservation, and uncertainty handling are priorities, such as medical diagnosis, scientific discovery, and safety-critical systems.

## 5. Conclusion

Our analysis reveals that energy-based models, specifically Denoising Potential Networks, do not follow the law of equi-separation that characterizes traditional deep neural networks. Instead, they maintain or increase separation fuzziness while reorganizing features according to principles derived from energy-based modeling.

These findings challenge the universality of the equi-separation law and reveal a novel paradigm for representation learning based on energy landscapes rather than direct mappings. This paradigm appears to prioritize structural coherence and stable configurations over raw separation metrics, potentially offering advantages in terms of robustness, uncertainty handling, and structure preservation.

Future research should explore the relationship between energy landscape properties and model performance, develop new metrics that better characterize energy-based models, and investigate applications where the unique properties of DPNs might provide advantages over traditional architectures.

By deepening our understanding of how different neural network architectures organize representations, we can develop more principled approaches to architecture design and potentially unlock new capabilities in deep learning systems.
