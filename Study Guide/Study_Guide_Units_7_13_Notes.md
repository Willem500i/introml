# Comprehensive Study Guide: Units 7-13
## Optimization, SVMs, Neural Networks, CNNs, PCA, Clustering, Decision Trees

*This guide covers all topics tested in Practice Tests 2 and 3. Use the blank sections to take your own notes.*

---

# PART 1: OPTIMIZATION & GRADIENT DESCENT (Unit 7)

## 1.1 Core Concept: What is Optimization?

**Definition**: Finding parameter values that minimize (or maximize) an objective function.

In machine learning, we typically want to **minimize a loss function** $J(\mathbf{w})$ with respect to parameters $\mathbf{w}$.

### My Notes:
```
[Space for your notes on when/why we need optimization]




```

---

## 1.2 Gradient Descent Algorithm

### The Update Rule

$$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \alpha \nabla J(\mathbf{w}^{(k)})$$

Where:
- $\mathbf{w}^{(k)}$ = current parameter values at iteration $k$
- $\alpha$ = **step size** (learning rate)
- $\nabla J(\mathbf{w}^{(k)})$ = gradient of the loss function at current point

### Intuition
The gradient $\nabla J$ points in the direction of **steepest ascent**. By moving in the **negative gradient direction**, we move toward lower loss values.

### My Notes on the Gradient Descent Update:
```
[Draw the gradient descent update on a simple function here]




```

---

## 1.3 Step Size (Learning Rate) Effects

| Step Size | Effect | What Happens |
|-----------|--------|--------------|
| **Too small** | Very slow convergence | Takes many iterations, may not reach minimum |
| **Too large** | Oscillation/divergence | Overshoots minimum, may never converge |
| **Just right** | Fast, stable convergence | Reaches minimum efficiently |

### Visualization:
```
Too small:       Just right:       Too large:
    •                •                  •
     \               \                 / \
      \               \               /   \
       \               ↓             /     ↘
        ↓              *            ↙       DIVERGE!
```

### My Notes on Learning Rate:
```
[When would you increase/decrease learning rate?]




```

---

## 1.4 Convergence Analysis (Quadratic Loss)

For a quadratic loss function:
$$J(\mathbf{w}) = \frac{1}{2}b_1 w_1^2 + \frac{1}{2}b_2 w_2^2$$

### Step-by-Step Derivation:

**Step 1: Compute the gradient**
$$\nabla J = \begin{pmatrix} b_1 w_1 \\ b_2 w_2 \end{pmatrix}$$

**Step 2: Write the update equations**
$$w_1^{(k+1)} = w_1^{(k)} - \alpha b_1 w_1^{(k)} = (1 - \alpha b_1) w_1^{(k)}$$
$$w_2^{(k+1)} = w_2^{(k)} - \alpha b_2 w_2^{(k)} = (1 - \alpha b_2) w_2^{(k)}$$

**Step 3: Convergence condition**
For convergence, we need $|1 - \alpha b_i| < 1$ for all $i$.

This gives us: $0 < \alpha < \frac{2}{b_{\max}}$

### Condition Number
$$\kappa = \frac{b_{\max}}{b_{\min}}$$

- **Large $\kappa$** → slow convergence (elongated elliptical contours)
- **Small $\kappa$** (close to 1) → fast convergence (circular contours)

### My Notes on Convergence:
```
[Work through why |1 - αb| < 1 is required]




```

---

## 1.5 Computing Gradients Using the Chain Rule

### General Chain Rule
For $J = J(z_1, z_2, ...)$ where each $z_i = z_i(w_1, w_2, ...)$:

$$\frac{\partial J}{\partial w_j} = \sum_i \frac{\partial J}{\partial z_i} \cdot \frac{\partial z_i}{\partial w_j}$$

### Worked Example
Given: $J = z_1 e^{z_1 z_2}$, where $z_1 = a_1 w_1 w_2$ and $z_2 = a_2 w_1 + a_3 w_2^2$

**Step 1: Partial derivatives of J w.r.t. intermediate variables**
$$\frac{\partial J}{\partial z_1} = e^{z_1 z_2} + z_1 z_2 e^{z_1 z_2} = e^{z_1 z_2}(1 + z_1 z_2)$$
$$\frac{\partial J}{\partial z_2} = z_1^2 e^{z_1 z_2}$$

**Step 2: Partial derivatives of intermediate variables w.r.t. parameters**
$$\frac{\partial z_1}{\partial w_1} = a_1 w_2, \quad \frac{\partial z_1}{\partial w_2} = a_1 w_1$$
$$\frac{\partial z_2}{\partial w_1} = a_2, \quad \frac{\partial z_2}{\partial w_2} = 2a_3 w_2$$

**Step 3: Apply chain rule**
$$\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial z_1}\frac{\partial z_1}{\partial w_1} + \frac{\partial J}{\partial z_2}\frac{\partial z_2}{\partial w_1}$$

### My Notes on Chain Rule:
```
[Practice: Given J = (z)^2, z = w₁w₂, find ∂J/∂w₁]




```

---

## 1.6 Local vs. Global Minima

| Type | Definition | When Guaranteed? |
|------|------------|------------------|
| **Local minimum** | Lowest point in a neighborhood | Always exists for bounded functions |
| **Global minimum** | Lowest point over entire domain | Guaranteed to be found for **convex** functions |

### Convex Functions
- Have only ONE minimum (local = global)
- Examples: Linear regression loss, logistic regression loss, ridge regression loss

### Non-Convex Functions
- May have multiple local minima
- Example: Neural network loss functions
- Gradient descent may get stuck in local minima

### My Notes on Convexity:
```
[Draw examples of convex vs. non-convex functions]




```

---

## 1.7 Testing Your Gradient Implementation

**Critical skill**: Always verify your gradient is correct!

### Numerical Gradient Check
For small $\epsilon$:
$$\nabla J(w) \approx \frac{J(w + \epsilon) - J(w - \epsilon)}{2\epsilon}$$

Or, check that:
$$J(w_1) - J(w_0) \approx \nabla J(w_0)^T (w_1 - w_0)$$

### My Notes:
```
[Why is gradient checking important before training?]




```

---

# PART 2: SUPPORT VECTOR MACHINES (Unit 8)

## 2.1 Linear Classifier Review

### Decision Rule
$$\hat{y} = \text{sign}(z), \quad z = \mathbf{w}^T \mathbf{x} + b$$

Where:
- $\mathbf{w}$ = weight vector (normal to decision boundary)
- $b$ = bias (offset)
- $\hat{y} \in \{-1, +1\}$

### Decision Boundary
The line (or hyperplane) where $\mathbf{w}^T \mathbf{x} + b = 0$

### My Notes on Linear Classifiers:
```
[Draw a 2D example with decision boundary]




```

---

## 2.2 Margin: The Key Concept

### Definition
The **margin** is the distance from the decision boundary to the nearest training point.

### Functional Margin vs. Geometric Margin

| Type | Formula | Interpretation |
|------|---------|----------------|
| **Functional margin** | $\gamma_i = y_i(\mathbf{w}^T \mathbf{x}_i + b)$ | How "confident" the prediction is |
| **Geometric margin** | $m = \frac{\min_i \gamma_i}{\|\mathbf{w}\|}$ | Actual distance to boundary |

### Why Maximize Margin?
- Larger margin → more robust to noise
- Better generalization to unseen data
- Points near boundary are most uncertain

### My Notes on Margin:
```
[Given w = [0, 2], b = -1.5, compute margin for a point]




```

---

## 2.3 Hard-Margin SVM

### Optimization Problem
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$
$$\text{subject to: } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i$$

### Key Points:
1. Minimizing $\|\mathbf{w}\|^2$ is equivalent to maximizing the margin
2. Constraint ensures all points are correctly classified with margin ≥ 1
3. **Only works if data is linearly separable**

### My Notes:
```
[Why is minimizing ||w||² the same as maximizing margin?]




```

---

## 2.4 Soft-Margin SVM (Hinge Loss)

### The Problem: What if Data Isn't Linearly Separable?

Introduce **slack variables** $\epsilon_i$ to allow some violations:

### Optimization Problem
$$\min_{\mathbf{w}, b, \boldsymbol{\epsilon}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \epsilon_i$$
$$\text{subject to: } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \epsilon_i, \quad \epsilon_i \geq 0$$

### Understanding Slack Variables

| $\epsilon_i$ value | Interpretation |
|--------------------|----------------|
| $\epsilon_i = 0$ | Correctly classified, outside margin |
| $0 < \epsilon_i < 1$ | Correctly classified, but inside margin |
| $\epsilon_i = 1$ | Exactly on decision boundary |
| $\epsilon_i > 1$ | Misclassified |

### The C Parameter
- **Large C**: Heavy penalty for violations → narrow margin, few violations
- **Small C**: Low penalty for violations → wide margin, more violations allowed

### Hinge Loss Formulation
$$\epsilon_i = \max(0, 1 - y_i z_i)$$

This is called the **hinge loss** because of its shape.

### My Notes on Soft Margin:
```
[Draw the hinge loss function]




```

---

## 2.5 Support Vectors

### Definition
**Support vectors** are the training points that:
- Lie exactly on the margin boundaries ($\epsilon_i = 0$ and $y_i z_i = 1$), OR
- Violate the margin ($\epsilon_i > 0$)

### Key Property
- Only support vectors affect the decision boundary
- Removing non-support vectors doesn't change the classifier
- Makes SVMs efficient for large datasets

### My Notes:
```
[In a scatter plot, which points would be support vectors?]




```

---

## 2.6 Kernel Trick

### The Problem
What if data isn't linearly separable in original space?

### The Solution
Map data to a higher-dimensional space where it becomes linearly separable:
$$\mathbf{x} \rightarrow \phi(\mathbf{x})$$

### The Kernel Trick
Instead of computing $\phi(\mathbf{x})$ explicitly, compute inner products directly:
$$K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$$

### Common Kernels

| Kernel | Formula | When to Use |
|--------|---------|-------------|
| **Linear** | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ | Linearly separable data |
| **Polynomial** | $K(\mathbf{x}, \mathbf{x}') = (1 + \mathbf{x}^T \mathbf{x}')^d$ | Polynomial decision boundaries |
| **RBF (Gaussian)** | $K(\mathbf{x}, \mathbf{x}') = e^{-\gamma \|\mathbf{x} - \mathbf{x}'\|^2}$ | Complex, nonlinear boundaries |

### The RBF Kernel
- $\gamma$ controls the "width" of influence
- Large $\gamma$: each point has small influence (can overfit)
- Small $\gamma$: each point has large influence (smoother boundary)

### My Notes on Kernels:
```
[How does RBF map to infinite dimensions?]




```

---

## 2.7 SVM Computation Example

### Given Data:
| $x_{i1}$ | $x_{i2}$ | $y_i$ |
|----------|----------|-------|
| 0 | 0 | -1 |
| 1 | 0.5 | -1 |
| 1 | 1 | +1 |
| 2 | 1 | +1 |

Classifier: $w_1 = 0$, $w_2 = 2$, $b = -1.5$

**Step 1: Compute z values**
- $z_1 = 0(0) + 2(0) - 1.5 = -1.5$
- $z_2 = 0(1) + 2(0.5) - 1.5 = -0.5$
- $z_3 = 0(1) + 2(1) - 1.5 = 0.5$
- $z_4 = 0(2) + 2(1) - 1.5 = 0.5$

**Step 2: Compute functional margins**
- $\gamma_1 = (-1)(-1.5) = 1.5$
- $\gamma_2 = (-1)(-0.5) = 0.5$
- $\gamma_3 = (+1)(0.5) = 0.5$
- $\gamma_4 = (+1)(0.5) = 0.5$

**Step 3: Compute geometric margin**
$$\|\mathbf{w}\| = \sqrt{0^2 + 2^2} = 2$$
$$m = \frac{\min_i \gamma_i}{\|\mathbf{w}\|} = \frac{0.5}{2} = 0.25$$

**Step 4: Identify support vectors**
Points 2, 3, 4 (minimum functional margin)

### My Practice Problem:
```
[Try a similar calculation with different w, b values]




```

---

# PART 3: NEURAL NETWORKS (Unit 9)

## 3.1 Single Neuron (Perceptron)

### Structure
$$z = \mathbf{w}^T \mathbf{x} + b$$
$$u = g(z)$$

Where $g$ is the **activation function**.

### My Notes:
```
[Draw a single neuron with inputs, weights, and output]




```

---

## 3.2 Activation Functions

### Why Activation Functions?
Without nonlinear activations, a multi-layer network would collapse to a single linear transformation!

### Common Activation Functions

| Function | Formula | Derivative | Pros/Cons |
|----------|---------|------------|-----------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | Smooth, but vanishing gradients |
| **ReLU** | $g(z) = \max(0, z)$ | $\mathbf{1}_{z>0}$ | Fast, but "dead" neurons |
| **Tanh** | $\tanh(z)$ | $1 - \tanh^2(z)$ | Zero-centered, but vanishing gradients |

### Sigmoid Properties (Important!)
- $\sigma(z) \in (0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### ReLU Advantages
1. No vanishing gradient (for $z > 0$)
2. Computationally simple
3. Sparse activations
4. Trains faster

### My Notes on Activations:
```
[Plot sigmoid, ReLU, and tanh side by side]




```

---

## 3.3 Single Hidden Layer Network

### Forward Propagation

**Step 1: Hidden layer pre-activation**
$$\mathbf{z}^H = W^H \mathbf{x} + \mathbf{b}^H$$

**Step 2: Hidden layer activation**
$$\mathbf{u}^H = g(\mathbf{z}^H)$$

**Step 3: Output layer**
$$z^O = \mathbf{w}^O \cdot \mathbf{u}^H + b^O$$

**Step 4: Output activation (if any)**
$$\hat{y} = g_{out}(z^O)$$

### Dimensions
- Input: $\mathbf{x} \in \mathbb{R}^d$
- Hidden weights: $W^H \in \mathbb{R}^{n_H \times d}$
- Hidden bias: $\mathbf{b}^H \in \mathbb{R}^{n_H}$
- Hidden activations: $\mathbf{u}^H \in \mathbb{R}^{n_H}$
- Output weights: $\mathbf{w}^O \in \mathbb{R}^{n_H}$

### My Notes on Forward Prop:
```
[Trace through a 2-input, 3-hidden, 1-output network]




```

---

## 3.4 Backpropagation

### The Key Idea
Compute gradients **layer by layer, from output to input** using the chain rule.

### Algorithm Steps

**Step 1: Compute loss gradient w.r.t. output**
For MSE loss $J = \frac{1}{2}(y - \hat{y})^2$:
$$\frac{\partial J}{\partial \hat{y}} = -(y - \hat{y})$$

**Step 2: Backprop through output layer**
$$\frac{\partial J}{\partial z^O} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^O}$$

If linear output: $\frac{\partial \hat{y}}{\partial z^O} = 1$

**Step 3: Gradient w.r.t. hidden activations**
$$\frac{\partial J}{\partial \mathbf{u}^H} = (W^O)^T \frac{\partial J}{\partial z^O}$$

**Step 4: Backprop through activation**
$$\frac{\partial J}{\partial \mathbf{z}^H} = \frac{\partial J}{\partial \mathbf{u}^H} \odot g'(\mathbf{z}^H)$$

**Step 5: Gradient w.r.t. weights**
$$\frac{\partial J}{\partial W^H} = \frac{\partial J}{\partial \mathbf{z}^H} \mathbf{x}^T$$

$$\frac{\partial J}{\partial \mathbf{b}^H} = \frac{\partial J}{\partial \mathbf{z}^H}$$

### My Notes on Backprop:
```
[Why is backprop more efficient than computing each gradient separately?]




```

---

## 3.5 Neural Network Computation Example

### Given:
- Input: $\mathbf{x} = [1, 0]^T$
- Hidden weights: $W^H = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$, $\mathbf{b}^H = \begin{pmatrix} 0 \\ -1 \end{pmatrix}$
- Output weights: $W^O = [1, 1]$, $b^O = 0$
- Activation: ReLU hidden, linear output

### Solution:

**Forward pass:**
1. $\mathbf{z}^H = W^H \mathbf{x} + \mathbf{b}^H = \begin{pmatrix} 1 \\ 1 \end{pmatrix} + \begin{pmatrix} 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

2. $\mathbf{u}^H = \text{ReLU}(\mathbf{z}^H) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

3. $z^O = W^O \mathbf{u}^H + b^O = [1, 1] \cdot [1, 0]^T + 0 = 1$

4. $\hat{y} = z^O = 1$

**Backward pass (if $y = 2$, MSE loss):**
1. $\frac{\partial J}{\partial \hat{y}} = -(2 - 1) = -1$

2. $\frac{\partial J}{\partial z^O} = -1 \cdot 1 = -1$

3. $\frac{\partial J}{\partial \mathbf{u}^H} = [1, 1]^T \cdot (-1) = [-1, -1]^T$

### My Practice Problem:
```
[Work through with different weights and sigmoid activation]




```

---

## 3.6 Number of Parameters

For a layer with $n_{in}$ inputs and $n_{out}$ outputs:
$$\text{Parameters} = n_{in} \times n_{out} + n_{out}$$

The $+n_{out}$ is for the biases.

### Example
Network: 784 inputs → 128 hidden → 10 outputs
- Layer 1: $784 \times 128 + 128 = 100,480$
- Layer 2: $128 \times 10 + 10 = 1,290$
- **Total: 101,770 parameters**

### My Notes:
```
[Calculate parameters for a 100→64→32→1 network]




```

---

# PART 4: CONVOLUTIONAL NEURAL NETWORKS (Unit 10)

## 4.1 Why CNNs for Images?

### Problems with Fully-Connected Layers for Images
- 256×256 RGB image = 196,608 inputs
- One FC layer with 1000 outputs = 196 million parameters!
- No spatial structure exploitation
- Not translation invariant

### CNN Advantages
1. **Parameter sharing**: Same filter applied everywhere
2. **Local connectivity**: Neurons only see local region
3. **Translation invariance**: Detect features anywhere
4. **Hierarchical features**: Low-level → high-level

### My Notes:
```
[Why does parameter sharing make sense for images?]




```

---

## 4.2 Convolution Operation

### 2D Convolution Formula
$$Z[i,j] = \sum_{k_1} \sum_{k_2} W[k_1, k_2] \cdot X[i+k_1, j+k_2]$$

### Output Size (Valid Convolution)
For input $(H, W)$ and kernel $(K_H, K_W)$:
- Output height: $H - K_H + 1$
- Output width: $W - K_W + 1$

### Multi-Channel Convolution
$$Z[i,j,m] = \sum_{k_1} \sum_{k_2} \sum_{n} W[k_1, k_2, n, m] \cdot X[i+k_1, j+k_2, n] + b[m]$$

Where:
- $n$ = input channel index
- $m$ = output channel (filter) index

### My Notes on Convolution:
```
[Work through a 3×3 convolution on a 5×5 input]




```

---

## 4.3 CNN Layer Parameters

### Number of Parameters
For kernel shape $(K_H, K_W, C_{in}, C_{out})$:
$$\text{Parameters} = K_H \times K_W \times C_{in} \times C_{out} + C_{out}$$

### Example Calculation
Input: $(32, 32, 3)$, Kernel: $(5, 5, 3, 16)$

**Parameters:**
- Weights: $5 \times 5 \times 3 \times 16 = 1200$
- Biases: $16$
- **Total: 1216**

**Output shape (valid):**
- $(32-5+1, 32-5+1, 16) = (28, 28, 16)$

**Multiplications:**
- Per output pixel: $5 \times 5 \times 3 = 75$
- Total output pixels: $28 \times 28 \times 16 = 12,544$
- **Total: $75 \times 12,544 = 940,800$**

### My Notes:
```
[Calculate for input (64, 64, 32), kernel (3, 3, 32, 64)]




```

---

## 4.4 Pooling Layers

### Purpose
1. Reduce spatial dimensions
2. Provide translation invariance
3. Reduce computation
4. Increase receptive field

### Types of Pooling

| Type | Formula | Use Case |
|------|---------|----------|
| **Max pooling** | $y[k] = \max_{j=0}^{p-1} x[sk+j]$ | Most common, keeps strongest features |
| **Average pooling** | $y[k] = \frac{1}{p}\sum_{j=0}^{p-1} x[sk+j]$ | Smoother, sometimes at output |

### Output Size After Pooling
For input $(H, W)$ with pool size $p$ and stride $s$:
- Output: $(\lfloor H/s \rfloor, \lfloor W/s \rfloor)$

### Example
Input: $(28, 28, 16)$, Max pool: size 2, stride 2
Output: $(14, 14, 16)$

### My Notes on Pooling:
```
[Why doesn't pooling have learnable parameters?]




```

---

## 4.5 Receptive Field

### Definition
The **receptive field** of a neuron is the region of the original input that can influence that neuron's output.

### Key Insight
- Deeper layers have larger receptive fields
- First layer: sees local patches
- Later layers: see larger, more global patterns

### My Notes:
```
[Calculate receptive field after two 3×3 conv layers]




```

---

## 4.6 Backpropagation in CNNs

### Through Activation
For $U = g(Z)$ with sigmoid:
$$\frac{\partial J}{\partial Z[i,j,m]} = \frac{\partial J}{\partial U[i,j,m]} \cdot U[i,j,m](1-U[i,j,m])$$

### Gradient w.r.t. Weights
$$\frac{\partial J}{\partial W[k_1,k_2,n,m]} = \sum_i \sum_j \frac{\partial J}{\partial Z[i,j,m]} \cdot X[i+k_1,j+k_2,n]$$

This is essentially a **convolution between input and gradient**!

### My Notes:
```
[Why is backprop in CNN also a convolution?]




```

---

# PART 5: PRINCIPAL COMPONENT ANALYSIS (Unit 11)

## 5.1 Goal of PCA

Find orthogonal directions that capture **maximum variance** in the data.

### Uses
1. **Dimensionality reduction**: Compress data
2. **Visualization**: Project high-dim data to 2D/3D
3. **Denoising**: Remove low-variance components
4. **Feature extraction**: Create uncorrelated features

### My Notes:
```
[When would you use PCA vs. other techniques?]




```

---

## 5.2 Computing PCA: Step by Step

### Algorithm

**Step 1: Center the data**
$$\tilde{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu}$$
where $\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$ (column means)

**Step 2: Compute covariance matrix**
$$Q = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$$

**Step 3: Eigendecomposition**
$$Q = V \Lambda V^T$$
where $\Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_d)$

**Step 4: Principal components**
- Columns of $V$ are the PCs (eigenvectors)
- $\lambda_j$ = variance explained by PC $j$
- Sort by descending eigenvalue

### Why Center the Data?
Without centering, the first PC would point toward the **mean** of the data rather than the direction of maximum variance.

### My Notes on PCA Algorithm:
```
[What does each eigenvalue represent?]




```

---

## 5.3 PCA Transformation

### Project to PC Coordinates
$$z_j = \mathbf{v}_j^T(\mathbf{x} - \boldsymbol{\mu})$$

Or in matrix form:
$$\mathbf{z} = V^T(\mathbf{x} - \boldsymbol{\mu})$$

### Reconstruct from k PCs
$$\hat{\mathbf{x}} = \boldsymbol{\mu} + \sum_{j=1}^k z_j \mathbf{v}_j = \boldsymbol{\mu} + V_k \mathbf{z}_k$$

### My Notes:
```
[Draw projection onto first PC in 2D]




```

---

## 5.4 Proportion of Variance Explained (PVE)

### Formula
$$\text{PVE}_k = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}$$

### Choosing Number of PCs
1. **PVE threshold**: Keep enough for PVE > 90% (or 95%)
2. **Scree plot**: Look for "elbow"
3. **Cross-validation**: If using for prediction

### My Notes:
```
[Given λ = [4.5, 2.1, 0.8, 0.6], how many PCs for 90%?]




```

---

## 5.5 Reconstruction Error

### Key Result
$$\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{j=k+1}^d z_j^2$$

The reconstruction error equals the sum of squared PC coefficients for the **skipped** components!

### Proof Sketch
1. Express centered data in PC basis: $\mathbf{x} - \boldsymbol{\mu} = \sum_{j=1}^d z_j \mathbf{v}_j$
2. Reconstruction uses only first k: $\hat{\mathbf{x}} - \boldsymbol{\mu} = \sum_{j=1}^k z_j \mathbf{v}_j$
3. Error: $\mathbf{x} - \hat{\mathbf{x}} = \sum_{j=k+1}^d z_j \mathbf{v}_j$
4. Use orthonormality of PCs to get squared norm

### My Notes:
```
[Why does orthonormality simplify the squared norm?]




```

---

## 5.6 PCA via SVD

### Connection
For centered data $\tilde{X}$, compute SVD:
$$\tilde{X} = U \Sigma V^T$$

Then:
- **Principal components**: columns of $V$
- **Singular values**: $\sigma_j = \sqrt{(n-1)\lambda_j}$
- **PC coefficients**: $Z = U\Sigma$

### Why Use SVD?
- Numerically more stable
- Can handle cases where $n < d$
- Efficient implementations available

### My Notes:
```
[Verify the relationship between σ² and λ]




```

---

## 5.7 PCA Computation Example

### Given Data:
$$X = \begin{pmatrix} 3 & 2 \\ 2 & 4 \\ 1 & 2 \\ 0 & 2 \end{pmatrix}$$

**Step 1: Sample mean**
$$\boldsymbol{\mu} = \frac{1}{4}[6, 10]^T = [1.5, 2.5]^T$$

**Step 2: Center data**
$$\tilde{X} = \begin{pmatrix} 1.5 & -0.5 \\ 0.5 & 1.5 \\ -0.5 & -0.5 \\ -1.5 & -0.5 \end{pmatrix}$$

**Step 3: Covariance matrix**
$$Q = \frac{1}{3}\tilde{X}^T\tilde{X} = \begin{pmatrix} 5/3 & 1/3 \\ 1/3 & 1 \end{pmatrix}$$

**Step 4: Eigenvalues** (from characteristic equation)
$$\lambda_1 \approx 1.805, \quad \lambda_2 \approx 0.862$$

**Step 5: PVE of first PC**
$$\text{PVE}_1 = \frac{1.805}{1.805 + 0.862} \approx 67.7\%$$

### My Practice:
```
[Work through eigenvalue calculation step by step]




```

---

# PART 6: CLUSTERING (Unit 12)

## 6.1 What is Clustering?

### Definition
**Unsupervised** grouping of data points based on similarity.

### Key Difference from Classification
- No labels provided
- Algorithm discovers natural groupings
- Number of clusters often unknown

### My Notes:
```
[When would you use clustering vs. classification?]




```

---

## 6.2 K-Means Algorithm

### Objective Function
$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

Minimize **within-cluster sum of squares** (WCSS).

### Algorithm

**Initialize**: Choose K initial centers $\boldsymbol{\mu}_1, ..., \boldsymbol{\mu}_K$

**Repeat until convergence:**

**Step 1 - Assignment**: Assign each point to nearest center
$$c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**Step 2 - Update**: Recompute centers
$$\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$$

### My Notes on K-Means:
```
[Why does each step decrease or maintain J?]




```

---

## 6.3 K-Means Convergence

### Theorem
K-means **always converges**.

### Proof Sketch
1. **Assignment step**: Each point moves to closer center → J decreases (or stays same)
2. **Update step**: Mean minimizes squared distances within cluster → J decreases (or stays same)
3. J is bounded below by 0
4. Finite number of possible assignments
5. Therefore, must eventually stop

### Important Caveats
- Converges to **local minimum**, not necessarily global
- Result depends on initialization
- Multiple runs with different initializations recommended

### My Notes:
```
[Why can K-means get stuck in local minima?]




```

---

## 6.4 Initialization Methods

### Random Initialization
- Choose K random data points as initial centers
- Simple but can give poor results

### K-Means++ (Better!)
1. Choose first center uniformly at random
2. For each remaining center:
   - Compute distance $D(x)$ to nearest existing center
   - Choose new center with probability $\propto D(x)^2$
3. Points far from existing centers more likely to be chosen

### My Notes:
```
[Why does K-means++ give better initialization?]




```

---

## 6.5 Choosing K

### Methods

| Method | Description |
|--------|-------------|
| **Elbow method** | Plot J vs K, find "elbow" where rate of decrease slows |
| **Silhouette score** | Measures cluster cohesion vs separation |
| **Gap statistic** | Compare J to null distribution |
| **Domain knowledge** | Use expected number of groups |

### Elbow Method Details
- J always decreases as K increases
- Look for point where additional clusters don't help much
- Can be subjective

### My Notes:
```
[Draw example elbow plot and identify the "elbow"]




```

---

## 6.6 K-Means Computation Example

### Given Points:
$(0,0), (1,0), (0,1), (3,3), (4,3)$

Initial centers: $(0,0)$ and $(3,3)$

**Step 1: Compute distances and assign**

| Point | Dist to (0,0) | Dist to (3,3) | Assign |
|-------|---------------|---------------|--------|
| (0,0) | 0 | $\sqrt{18}$ ≈ 4.24 | C1 |
| (1,0) | 1 | $\sqrt{13}$ ≈ 3.61 | C1 |
| (0,1) | 1 | $\sqrt{13}$ ≈ 3.61 | C1 |
| (3,3) | $\sqrt{18}$ ≈ 4.24 | 0 | C2 |
| (4,3) | 5 | 1 | C2 |

**Step 2: Update centers**
$$\boldsymbol{\mu}_1^{new} = \frac{1}{3}[(0,0) + (1,0) + (0,1)] = (0.33, 0.33)$$
$$\boldsymbol{\mu}_2^{new} = \frac{1}{2}[(3,3) + (4,3)] = (3.5, 3)$$

**Step 3: Check convergence**
Centers changed → Not converged, continue...

### My Practice:
```
[Do one more iteration of K-means]




```

---

# PART 7: DECISION TREES (Unit 13)

## 7.1 Decision Tree Structure

### Components
- **Root node**: Top of tree, first split
- **Internal nodes**: Feature tests (splits)
- **Leaf nodes**: Final predictions
- **Edges**: Outcomes of tests

### How Predictions Work
1. Start at root
2. At each node, test a feature condition
3. Follow corresponding branch
4. Repeat until reaching a leaf
5. Output leaf's prediction

### My Notes:
```
[Draw a simple decision tree for "play tennis" decision]




```

---

## 7.2 Splitting Criteria for Classification

### Goal
Find splits that create **pure** (homogeneous) child nodes.

### Impurity Measures

| Measure | Formula | Range |
|---------|---------|-------|
| **Misclassification rate** | $1 - \max_k p_k$ | [0, 1-1/K] |
| **Gini impurity** | $\sum_k p_k(1-p_k) = 1 - \sum_k p_k^2$ | [0, 1-1/K] |
| **Entropy** | $-\sum_k p_k \log_2(p_k)$ | [0, log₂(K)] |

Where $p_k$ = proportion of samples in class $k$.

### Properties
- All equal 0 for pure nodes
- Maximum when classes equally distributed
- Gini and Entropy similar in practice
- Gini slightly faster to compute

### My Notes on Impurity:
```
[Calculate Gini for p = [0.5, 0.5] and p = [0.9, 0.1]]




```

---

## 7.3 Information Gain

### Definition
Reduction in impurity from a split:
$$\text{IG} = H(\text{parent}) - \sum_{\text{children}} \frac{n_{\text{child}}}{n_{\text{parent}}} H(\text{child})$$

### Example Calculation
Parent: 2 samples of class 0, 2 of class 1
- $\text{Gini}_{\text{parent}} = 2(0.5)(0.5) = 0.5$

Split on $x_1 = 0.5$:
- Left child: 2 samples, all class 0 → $\text{Gini} = 0$
- Right child: 2 samples, all class 1 → $\text{Gini} = 0$

$$\text{IG} = 0.5 - \left(\frac{2}{4}(0) + \frac{2}{4}(0)\right) = 0.5$$

This is a **perfect split**!

### My Notes:
```
[When is information gain maximized?]




```

---

## 7.4 Overfitting in Decision Trees

### The Problem
Deep trees can fit training data perfectly but:
- Many leaves with few samples
- Capture noise in data
- Poor generalization

### Signs of Overfitting
- Training accuracy >> Test accuracy
- Very deep tree
- Leaves with single samples

### My Notes:
```
[Why are deep trees prone to overfitting?]




```

---

## 7.5 Preventing Overfitting: Pruning

### Pre-Pruning (Early Stopping)
Stop splitting when:
- Maximum depth reached
- Minimum samples per leaf
- Minimum samples to split
- Minimum information gain

### Post-Pruning
1. Grow full tree
2. Evaluate removing each split
3. Remove splits that don't improve validation error
4. Continue until no improvement

### Hyperparameter Tuning
Use cross-validation to select:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### My Notes:
```
[Trade-offs between pre-pruning and post-pruning?]




```

---

## 7.6 Regression Trees

### Splitting Criterion
Minimize **mean squared error** within each region:
$$\text{MSE} = \frac{1}{n}\sum_{i \in \text{node}}(y_i - \bar{y})^2$$

### Prediction
Each leaf outputs the **mean** of training samples in that region.

### My Notes:
```
[How is splitting in regression trees similar to classification?]




```

---

## 7.7 Random Forests

### The Problem with Single Trees
- High variance
- Sensitive to training data
- Easily overfit

### Random Forest Solution
Combine many trees through **ensemble averaging**:

### Algorithm
1. Create B bootstrap samples (sampling with replacement)
2. For each sample, train a decision tree:
   - At each split, consider random subset of features
3. Aggregate predictions:
   - **Regression**: Average
   - **Classification**: Majority vote

### Why It Works
- **Bagging** reduces variance
- **Random feature selection** decorrelates trees
- Individual trees can overfit; ensemble generalizes better

### Key Hyperparameters
- Number of trees (more is usually better)
- Features to consider at each split (typically $\sqrt{d}$)
- Individual tree settings

### My Notes on Random Forests:
```
[Why does averaging many trees reduce variance?]




```

---

## 7.8 Decision Tree Computation Example

### Data:
| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

**Root node Gini:**
$p_0 = 0.5$, $p_1 = 0.5$
$$\text{Gini} = 2(0.5)(0.5) = 0.5$$

**Split on $x_1 = 0.5$:**

Left ($x_1 < 0.5$): both y=0 → Gini = 0
Right ($x_1 \geq 0.5$): both y=1 → Gini = 0

**Weighted Gini after split:**
$$\frac{2}{4}(0) + \frac{2}{4}(0) = 0$$

**Information Gain:**
$$\text{IG} = 0.5 - 0 = 0.5$$

This is the maximum possible gain!

### My Practice:
```
[What if we split on x₂ instead?]




```

---

# SUMMARY CHECKLIST

## Before taking the practice tests, make sure you can:

### Unit 7: Optimization
- [ ] Write the gradient descent update rule
- [ ] Explain effects of step size
- [ ] Compute gradients using chain rule
- [ ] Determine convergence conditions for quadratic
- [ ] Distinguish local vs. global minima

### Unit 8: SVMs
- [ ] Compute functional and geometric margins
- [ ] Identify support vectors
- [ ] Explain hard vs. soft margin
- [ ] Write the hinge loss
- [ ] Explain the kernel trick

### Unit 9: Neural Networks
- [ ] Perform forward propagation
- [ ] Compute derivatives of activation functions
- [ ] Perform backpropagation
- [ ] Calculate number of parameters

### Unit 10: CNNs
- [ ] Compute output dimensions after convolution
- [ ] Calculate number of parameters
- [ ] Explain purpose of pooling
- [ ] Define receptive field

### Unit 11: PCA
- [ ] Center data and compute covariance
- [ ] Find eigenvalues and eigenvectors
- [ ] Calculate PVE
- [ ] Project and reconstruct data
- [ ] Relate PCA to SVD

### Unit 12: Clustering
- [ ] Write K-means objective
- [ ] Perform K-means iterations by hand
- [ ] Explain convergence guarantee
- [ ] Describe methods for choosing K

### Unit 13: Decision Trees
- [ ] Calculate Gini impurity
- [ ] Compute information gain
- [ ] Explain overfitting and pruning
- [ ] Describe Random Forest algorithm

---

# FORMULA QUICK REFERENCE

## Gradient Descent
$$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \alpha \nabla J(\mathbf{w}^{(k)})$$

## SVM Margin
$$m = \frac{\min_i y_i(\mathbf{w}^T \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

## Hinge Loss
$$\epsilon_i = \max(0, 1 - y_i z_i)$$

## Sigmoid
$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z) = \sigma(z)(1-\sigma(z))$$

## ReLU
$$g(z) = \max(0, z), \quad g'(z) = \mathbf{1}_{z > 0}$$

## Conv Output Size
$$H_{out} = H_{in} - K_H + 1$$

## Conv Parameters
$$\text{Params} = K_H \times K_W \times C_{in} \times C_{out} + C_{out}$$

## Covariance Matrix
$$Q = \frac{1}{n-1}\tilde{X}^T\tilde{X}$$

## PVE
$$\text{PVE}_k = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}$$

## K-Means Objective
$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

## Gini Impurity
$$\text{Gini} = \sum_k p_k(1-p_k) = 1 - \sum_k p_k^2$$

---

*Good luck with your studying! Work through the examples, fill in your notes, and then tackle the practice tests.*

