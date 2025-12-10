# Introduction to Machine Learning - Complete Study Guide

## Table of Contents
1. [Unit 1: Introduction to ML](#unit-1-introduction-to-ml)
2. [Unit 2: Simple Linear Regression](#unit-2-simple-linear-regression)
3. [Unit 3: Multiple Linear Regression](#unit-3-multiple-linear-regression)
4. [Unit 4: Model Selection & Cross-Validation](#unit-4-model-selection--cross-validation)
5. [Unit 5: LASSO & Regularization](#unit-5-lasso--regularization)
6. [Unit 6: Logistic Regression](#unit-6-logistic-regression)
7. [Unit 7: Optimization & Gradient Descent](#unit-7-optimization--gradient-descent)
8. [Unit 8: Support Vector Machines](#unit-8-support-vector-machines)
9. [Unit 9: Neural Networks](#unit-9-neural-networks)
10. [Unit 10: Convolutional Neural Networks](#unit-10-convolutional-neural-networks)
11. [Unit 11: Principal Component Analysis](#unit-11-principal-component-analysis)
12. [Unit 12: Clustering](#unit-12-clustering)
13. [Unit 13: Decision Trees](#unit-13-decision-trees)

---

# Unit 1: Introduction to ML

## Key Concepts

### Supervised vs Unsupervised Learning
- **Supervised Learning**: Learn from labeled data (input-output pairs)
  - Regression: continuous target variable
  - Classification: discrete target variable
- **Unsupervised Learning**: Find patterns in unlabeled data
  - Clustering, dimensionality reduction

### Key Terminology
- **Features/Predictors** (X): Input variables
- **Target/Response** (y): Output variable to predict
- **Training Data**: Data used to fit the model
- **Test Data**: Data used to evaluate model performance
- **Model Parameters**: Values learned from data (e.g., β in linear regression)
- **Hyperparameters**: Values set before training (e.g., regularization λ)

### The ML Pipeline
1. Data collection and preprocessing
2. Feature engineering
3. Model selection
4. Training (fitting)
5. Validation and hyperparameter tuning
6. Testing and evaluation
7. Deployment

---

# Unit 2: Simple Linear Regression

## Key Concepts

### The Model
$$y = \beta_0 + \beta_1 x + \epsilon$$

where:
- $\beta_0$: intercept
- $\beta_1$: slope
- $\epsilon$: error term (noise)

### Least Squares Estimation
**Objective**: Minimize the Residual Sum of Squares (RSS)
$$RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$$

**Closed-form solution**:
$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{s_{xy}}{s_x^2}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

### Important Formulas
- **Sample mean**: $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$
- **Sample variance**: $s_x^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$
- **Sample covariance**: $s_{xy} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$
- **Correlation**: $r = \frac{s_{xy}}{s_x s_y}$

### Model Without Intercept
For model $y = \beta x$:
$$\hat{\beta} = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2}$$

### Transforming Non-Linear Models
**Example**: Exponential decay $z(t) = z_0 e^{-\alpha t}$

Take log: $\ln(z) = \ln(z_0) - \alpha t$

Let $y = \ln(z)$, $\beta_0 = \ln(z_0)$, $\beta_1 = -\alpha$

Then fit: $y = \beta_0 + \beta_1 t$

---

# Unit 3: Multiple Linear Regression

## Key Concepts

### The Model
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$$

**Matrix form**: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$

where:
$$\mathbf{X} = \begin{pmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix}$$

### Least Squares Solution
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**Normal Equations**: $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$

### Transformed Linear Models (Basis Functions)
Any model of the form:
$$\hat{y} = \sum_{j=0}^{p} \beta_j \phi_j(\mathbf{x})$$

is linear in the parameters $\beta_j$ and can be solved with least squares.

**Examples**:
- Polynomial: $\phi_j(x) = x^j$
- Interaction terms: $\phi(x_1, x_2) = x_1 x_2$
- Indicator functions for piecewise models

### Feature Matrix Construction
Given basis functions $\phi_1, \phi_2, \ldots, \phi_p$:
$$A_{ij} = \phi_j(\mathbf{x}_i)$$

Then: $\hat{\boldsymbol{\beta}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{y}$

---

# Unit 4: Model Selection & Cross-Validation

## Key Concepts

### Bias-Variance Tradeoff
$$\text{Expected Test Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

- **Bias**: Error from approximating a complex problem with a simple model (underfitting)
- **Variance**: Error from sensitivity to training data fluctuations (overfitting)

### Under-modeling
When the true function $f_0(\mathbf{x})$ cannot be expressed by the model class $f(\mathbf{x}, \boldsymbol{\beta})$.

**Example**: True function is quadratic $f_0(x) = 1 + 2x + x^2$, but fitting linear $f(x) = \beta_0 + \beta_1 x$

### Training Error vs Test Error
- **Training Error**: Always decreases with model complexity
- **Test Error**: Decreases then increases (U-shaped curve)
- **Goal**: Minimize test error, not training error

### Cross-Validation
**K-fold Cross-Validation**:
1. Split data into K equal folds
2. For each fold k:
   - Train on all folds except k
   - Test on fold k
3. Average the K test errors

**Leave-One-Out CV (LOOCV)**: K = n (each sample is its own fold)

### One Standard Error Rule
1. Find the model with lowest CV error
2. Compute standard error of CV error
3. Select simplest model within one SE of the minimum

### Estimating Bias
$$\text{Bias}(x) = \mathbb{E}[f(x, \hat{\boldsymbol{\beta}})] - f_0(x)$$

For no noise in training data: $\text{Bias} = 0$ if model has no under-modeling

---

# Unit 5: LASSO & Regularization

## Key Concepts

### Ridge Regression (L2 Regularization)
$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$

**Closed-form solution**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

### LASSO (L1 Regularization)
$$\hat{\boldsymbol{\beta}}_{\text{LASSO}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

**Key property**: LASSO performs feature selection (sets some $\beta_j = 0$)

### Soft Thresholding (1D Case)
For scalar problem: $\hat{w} = \arg\min_w \frac{1}{2}(y-w)^2 + \lambda|w|$

**Solution (Soft thresholding)**:
$$\hat{w} = \begin{cases} y - \lambda & \text{if } y > \lambda \\ 0 & \text{if } |y| \leq \lambda \\ y + \lambda & \text{if } y < -\lambda \end{cases}$$

### Normalization Before LASSO
**Critical**: Features must be normalized to have same scale, otherwise LASSO penalizes different features unequally.

$$z_j = \frac{x_j - \bar{x}_j}{s_j}, \quad u = \frac{y - \bar{y}}{s_y}$$

### Converting Back to Original Scale
If $\hat{u} = \sum_j \alpha_j z_j$, then in original coordinates:
$$\hat{y} = \bar{y} + s_y \sum_j \alpha_j \frac{x_j - \bar{x}_j}{s_j} = \beta_0 + \sum_j \beta_j x_j$$

where $\beta_j = s_y \alpha_j / s_j$ and $\beta_0 = \bar{y} - \sum_j \beta_j \bar{x}_j$

---

# Unit 6: Logistic Regression

## Key Concepts

### Binary Classification
**Goal**: Predict $y \in \{0, 1\}$ or $y \in \{-1, +1\}$

### Logistic Function (Sigmoid)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties**:
- $\sigma(z) \in (0, 1)$
- $\sigma(0) = 0.5$
- $\sigma(-z) = 1 - \sigma(z)$
- $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

### Logistic Regression Model
$$P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

**Decision boundary**: Predict $\hat{y} = 1$ if $z > 0$, else $\hat{y} = 0$

### Log-Odds (Logit)
$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = z = \boldsymbol{\beta}^T \mathbf{x}$$

### Binary Cross-Entropy Loss
$$J(\boldsymbol{\beta}) = -\sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$

Or equivalently (for computation):
$$J(\boldsymbol{\beta}) = \sum_{i=1}^N \left[ \log(1 + e^{z_i}) - y_i z_i \right]$$

### Gradients for Logistic Regression
$$\frac{\partial J}{\partial \beta_j} = \sum_{i=1}^N (\sigma(z_i) - y_i) x_{ij}$$

**No closed-form solution** - must use iterative optimization (gradient descent)

---

# Unit 7: Optimization & Gradient Descent

## Key Concepts

### Gradient Descent
**Update rule**:
$$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \alpha \nabla J(\mathbf{w}^{(k)})$$

where $\alpha$ is the **step size** (learning rate)

### Computing Gradients (Chain Rule)
For composite functions, use chain rule:
$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Convergence
For quadratic: $J(\mathbf{w}) = \frac{1}{2}b_1 w_1^2 + \frac{1}{2}b_2 w_2^2$

- Gradient: $\nabla J = (b_1 w_1, b_2 w_2)$
- Update: $w_i^{(k+1)} = (1 - \alpha b_i) w_i^{(k)}$
- Converges if $|1 - \alpha b_i| < 1 \Rightarrow 0 < \alpha < 2/b_{\max}$

### Condition Number
$$\kappa = \frac{b_{\max}}{b_{\min}}$$

**Large condition number** → slow convergence

### Local vs Global Minima
- Gradient descent can get stuck in local minima
- For convex functions, local minimum = global minimum
- Non-convex functions (neural networks) may have many local minima

### Gradient Calculations Examples

**Example 1**: $J = z_1 e^{z_1 z_2}$, $z_1 = a_1 w_1 w_2$, $z_2 = a_2 w_1 + a_3 w_2^2$

$$\frac{\partial J}{\partial z_1} = e^{z_1 z_2}(1 + z_1 z_2)$$
$$\frac{\partial J}{\partial z_2} = z_1^2 e^{z_1 z_2}$$
$$\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial z_1}\frac{\partial z_1}{\partial w_1} + \frac{\partial J}{\partial z_2}\frac{\partial z_2}{\partial w_1}$$

---

# Unit 8: Support Vector Machines

## Key Concepts

### Linear Classifier
$$\hat{y} = \text{sign}(b + \mathbf{w}^T \mathbf{x})$$

where $\hat{y} \in \{-1, +1\}$

### Margin
The margin is the distance from the decision boundary to the nearest training point:
$$m = \frac{\gamma}{\|\mathbf{w}\|}$$

where $\gamma = \min_i y_i(b + \mathbf{w}^T \mathbf{x}_i)$

### Maximum Margin Classifier
$$\max_{\mathbf{w}, b} \frac{1}{\|\mathbf{w}\|} \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \; \forall i$$

Equivalent to: $\min_{\mathbf{w}, b} \|\mathbf{w}\|^2$ s.t. constraints

### Soft Margin SVM (Hinge Loss)
$$\min_{\mathbf{w}, b, \boldsymbol{\epsilon}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^n \epsilon_i$$
$$\text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \epsilon_i, \quad \epsilon_i \geq 0$$

**Hinge Loss**: $\epsilon_i = \max(0, 1 - y_i z_i)$

- $\epsilon_i = 0$: correctly classified, outside margin
- $0 < \epsilon_i < 1$: correctly classified, inside margin
- $\epsilon_i > 1$: misclassified

### Support Vectors
Points that lie on the margin boundaries ($\epsilon_i = 0$) or violate the margin ($\epsilon_i > 0$)

### Kernel Trick
Replace inner products with kernel functions:
$$z = \sum_{i} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x})$$

**Common kernels**:
- Linear: $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$
- Polynomial: $K(\mathbf{x}, \mathbf{x}') = (1 + \mathbf{x}^T \mathbf{x}')^d$
- RBF (Gaussian): $K(\mathbf{x}, \mathbf{x}') = e^{-\gamma \|\mathbf{x} - \mathbf{x}'\|^2}$

---

# Unit 9: Neural Networks

## Key Concepts

### Single Hidden Layer Network
**Forward propagation**:
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
$$\mathbf{u}^{(1)} = g(\mathbf{z}^{(1)})$$
$$z^{(2)} = \mathbf{w}^{(2)T} \mathbf{u}^{(1)} + b^{(2)}$$
$$\hat{y} = g_{out}(z^{(2)})$$

### Activation Functions
- **ReLU**: $g(z) = \max(0, z)$, derivative: $g'(z) = \mathbf{1}_{z > 0}$
- **Sigmoid**: $g(z) = \frac{1}{1+e^{-z}}$, derivative: $g'(z) = g(z)(1-g(z))$
- **Tanh**: $g(z) = \tanh(z)$, derivative: $g'(z) = 1 - g(z)^2$
- **Threshold**: $g(z) = \mathbf{1}_{z \geq 0}$ (not differentiable)

### Loss Functions
- **MSE (regression)**: $J = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$
- **Cross-entropy (classification)**: $J = -\sum_{i=1}^N [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$

### Backpropagation
**Key idea**: Compute gradients layer by layer, from output to input

For a loss $J$ and layer with output $\mathbf{u} = g(\mathbf{z})$:

1. Given $\frac{\partial J}{\partial \mathbf{u}}$
2. Compute $\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{u}} \odot g'(\mathbf{z})$ (element-wise)
3. Compute $\frac{\partial J}{\partial \mathbf{W}} = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^T$
4. Compute $\frac{\partial J}{\partial \mathbf{x}} = \mathbf{W}^T \frac{\partial J}{\partial \mathbf{z}}$ (for backprop to previous layer)

### Number of Parameters
For a layer with $n_{in}$ inputs and $n_{out}$ outputs:
$$\text{Parameters} = n_{in} \times n_{out} + n_{out}$$
(weights + biases)

---

# Unit 10: Convolutional Neural Networks

## Key Concepts

### Tensor Shapes
- **Images**: (batch, height, width, channels)
- **1D signals**: (batch, time, channels)
- **Videos**: (batch, frames, height, width, channels)

### 2D Convolution
$$Z[i,j] = \sum_{k_1} \sum_{k_2} W[k_1, k_2] X[i+k_1, j+k_2]$$

**Output size** (valid convolution):
- Input: $H \times W$
- Kernel: $K_H \times K_W$
- Output: $(H - K_H + 1) \times (W - K_W + 1)$

### Multi-channel Convolution
$$Z[i,j,m] = \sum_{k_1} \sum_{k_2} \sum_{n} W[k_1, k_2, n, m] X[i+k_1, j+k_2, n] + b[m]$$

where $n$ = input channel, $m$ = output channel

### Number of Parameters
For kernel shape $(K_H, K_W, C_{in}, C_{out})$:
$$\text{Parameters} = K_H \times K_W \times C_{in} \times C_{out} + C_{out}$$

### Pooling
**Max pooling**: $y[k] = \max_{j=0}^{p-1} x[sk+j]$
**Average pooling**: $y[k] = \frac{1}{p}\sum_{j=0}^{p-1} x[sk+j]$

where $s$ = stride, $p$ = pool size

### Backpropagation in CNNs
For $U = g(Z)$ with sigmoid activation:
$$\frac{\partial J}{\partial Z} = \frac{\partial J}{\partial U} \odot U \odot (1-U)$$

For weights:
$$\frac{\partial J}{\partial W[k_1,k_2,n,m]} = \sum_i \sum_{j_1} \sum_{j_2} \frac{\partial J}{\partial Z[i,j_1,j_2,m]} X[i,j_1+k_1,j_2+k_2,n]$$

---

# Unit 11: Principal Component Analysis (PCA)

## Key Concepts

### Goal
Find orthogonal directions that capture maximum variance in the data.

### Computing PCA
1. **Center the data**: $\tilde{\mathbf{X}} = \mathbf{X} - \boldsymbol{\mu}$
2. **Compute covariance matrix**: $\mathbf{Q} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$
3. **Eigendecomposition**: $\mathbf{Q} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$
4. **Principal components**: columns of $\mathbf{V}$ (eigenvectors)
5. **Eigenvalues** $\lambda_i$: variance explained by each PC

### PCA Transform
**Project to PC coordinates**: $\mathbf{z}_i = \mathbf{V}^T(\mathbf{x}_i - \boldsymbol{\mu})$

**Reconstruct from $k$ PCs**:
$$\hat{\mathbf{x}}_i = \boldsymbol{\mu} + \sum_{j=1}^k z_{ij} \mathbf{v}_j$$

### Proportion of Variance Explained
$$\text{PVE}_k = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}$$

### Reconstruction Error
$$\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{j=k+1}^d z_j^2$$

Sum of squared PC coefficients for skipped components.

### SVD for PCA
For centered data $\tilde{\mathbf{X}} = \mathbf{U}\mathbf{S}\mathbf{V}^T$:
- PCs: columns of $\mathbf{V}$
- Singular values: $s_j = \sqrt{(n-1)\lambda_j}$
- PC coefficients: $\mathbf{Z} = \mathbf{U}\mathbf{S}$

---

# Unit 12: Clustering

## Key Concepts

### K-Means Algorithm
1. **Initialize** K cluster centers $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$
2. **Assign** each point to nearest center:
   $$c_i = \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
3. **Update** centers:
   $$\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$$
4. **Repeat** steps 2-3 until convergence

### K-Means Objective
$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

### Initialization Methods
- **Random**: Choose K random samples as initial centers
- **K-means++**: Sequential selection with probability proportional to squared distance

### Computational Complexity
- Per iteration: $O(NKD)$ where N = samples, K = clusters, D = dimensions
- Total: $O(TNKD)$ for T iterations

### Choosing K
- **Elbow method**: Plot J vs K, find "elbow"
- **Silhouette score**: Measure cluster cohesion vs separation
- **Gap statistic**: Compare to null distribution

---

# Unit 13: Decision Trees

## Key Concepts

### Structure
- **Internal nodes**: feature tests (splits)
- **Leaf nodes**: predictions
- **Edges**: outcomes of tests

### Splitting Criteria
**For classification**:
- **Misclassification rate**: $1 - \max_k p_k$
- **Gini impurity**: $\sum_k p_k(1-p_k) = 1 - \sum_k p_k^2$
- **Entropy**: $-\sum_k p_k \log p_k$

**For regression**:
- **MSE**: $\frac{1}{n}\sum_{i \in \text{node}}(y_i - \bar{y})^2$

### Information Gain
$$IG = H(\text{parent}) - \sum_{\text{children}} \frac{n_{\text{child}}}{n_{\text{parent}}} H(\text{child})$$

### Pruning
- **Pre-pruning**: Stop splitting early (min samples, max depth)
- **Post-pruning**: Grow full tree, then remove nodes that don't improve validation error

### Random Forests
1. Create B bootstrap samples
2. For each sample, train a tree using random subset of features at each split
3. Average predictions (regression) or majority vote (classification)

**Benefits**: Reduces overfitting, handles high-dimensional data

---

# Key Formulas Quick Reference

## Linear Regression
- Least squares: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

## Regularization
- Ridge: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$
- LASSO: No closed form, use coordinate descent

## Logistic Regression
- $P(y=1|\mathbf{x}) = \sigma(\boldsymbol{\beta}^T\mathbf{x})$
- Gradient: $\nabla J = \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$

## SVM
- Margin: $m = \frac{1}{\|\mathbf{w}\|}$
- Hinge loss: $\max(0, 1 - y \cdot z)$

## Neural Networks
- ReLU: $g(z) = \max(0, z)$
- Sigmoid: $g(z) = 1/(1+e^{-z})$

## PCA
- Variance explained: $\lambda_j$
- Reconstruction: $\hat{\mathbf{x}} = \boldsymbol{\mu} + \mathbf{V}_k \mathbf{z}$

---

# Common Proof Patterns

## Deriving Least Squares
1. Write RSS: $J(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2$
2. Expand: $J = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$
3. Take gradient: $\nabla J = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$
4. Set to zero: $\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$
5. Solve: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

## Computing Gradients
1. Identify intermediate variables
2. Apply chain rule: $\frac{\partial J}{\partial w} = \sum_i \frac{\partial J}{\partial z_i}\frac{\partial z_i}{\partial w}$
3. For matrices: be careful with dimensions

## Showing Bias = 0
1. Write $\hat{\boldsymbol{\beta}}$ in terms of data
2. Substitute true model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta}_0 + \boldsymbol{\epsilon}$
3. Take expectation $\mathbb{E}[\hat{\boldsymbol{\beta}}]$
4. If $\mathbb{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}_0$, then unbiased

