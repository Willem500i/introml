# Comprehensive Final Practice Test

**Time Estimate**: 2.5-3 hours
**Total Points**: 150
**Coverage**: All Units (1-13)

---

## Section A: Short Answer Concepts (40 points)

*Answer each question in 2-4 sentences.*

### Q1 (3 pts)
What is the bias-variance tradeoff? How does model complexity affect each component?

<details>
<summary><strong>Solution</strong></summary>

The bias-variance tradeoff states that expected test error = Bias² + Variance + Irreducible Error. As model complexity increases, bias decreases (model can fit more patterns) but variance increases (model becomes sensitive to training data fluctuations). The optimal complexity balances these to minimize total error.
</details>

### Q2 (3 pts)
Why does LASSO produce sparse solutions while Ridge regression does not?

<details>
<summary><strong>Solution</strong></summary>

LASSO uses L1 regularization (sum of absolute values), which has a diamond-shaped constraint region with corners at the axes. The RSS contours often first touch this region at the corners, where some coefficients are exactly zero. Ridge uses L2 (sum of squares), which has a circular constraint region with no corners—solutions typically have all non-zero coefficients.
</details>

### Q3 (3 pts)
What is the "kernel trick" in SVMs and why is it useful?

<details>
<summary><strong>Solution</strong></summary>

The kernel trick allows SVMs to compute inner products in high-dimensional feature spaces without explicitly computing the feature transformation. This enables learning nonlinear decision boundaries efficiently—data that isn't linearly separable in original space may become separable in the implicit higher-dimensional space. Common kernels include RBF and polynomial.
</details>

### Q4 (3 pts)
Explain backpropagation in neural networks. Why is it efficient?

<details>
<summary><strong>Solution</strong></summary>

Backpropagation computes gradients of the loss with respect to all network parameters by applying the chain rule layer by layer, from output to input. It's efficient because it reuses intermediate computations—each layer's gradients are computed once and passed backward. The complexity is O(number of weights) per sample, not exponential.
</details>

### Q5 (3 pts)
What is the purpose of pooling layers in CNNs?

<details>
<summary><strong>Solution</strong></summary>

Pooling layers reduce spatial dimensions of feature maps, which decreases the number of parameters and computation in subsequent layers. They also provide translation invariance (small shifts in input don't change output much) and increase the receptive field of subsequent layers, allowing them to capture more global patterns.
</details>

### Q6 (3 pts)
How do you interpret eigenvalues in PCA? What do they represent?

<details>
<summary><strong>Solution</strong></summary>

Eigenvalues of the covariance matrix represent the variance of the data along each principal component direction. Larger eigenvalues indicate directions that capture more of the data's variability. The proportion of variance explained by the first k PCs is λ₁+...+λₖ divided by the sum of all eigenvalues.
</details>

### Q7 (3 pts)
What are the two steps in each K-means iteration? Why does the algorithm converge?

<details>
<summary><strong>Solution</strong></summary>

K-means alternates between (1) Assignment: assign each point to its nearest cluster center, and (2) Update: recompute each center as the mean of its assigned points. The algorithm converges because each step decreases (or maintains) the within-cluster sum of squares objective, and there are finitely many possible assignments.
</details>

### Q8 (4 pts)
Compare and contrast: misclassification rate, Gini impurity, and entropy as splitting criteria for decision trees.

<details>
<summary><strong>Solution</strong></summary>

All three measure node impurity—how mixed the class labels are:
- **Misclassification rate**: 1 - max(pₖ). Simple but not differentiable; doesn't distinguish between nodes with same majority class.
- **Gini impurity**: Σpₖ(1-pₖ). Measures probability of wrong random assignment. Tends to isolate most common class.
- **Entropy**: -Σpₖlog(pₖ). Information-theoretic measure. More sensitive to changes in probability distribution.

Gini and entropy are more commonly used because they're differentiable and encourage purer splits.
</details>

### Q9 (4 pts)
What is the one standard error rule for model selection? When and why would you use it?

<details>
<summary><strong>Solution</strong></summary>

The one standard error rule says: select the simplest model whose cross-validation error is within one standard error of the minimum. You'd use it when interpretability or robustness is important. Rationale: the "best" model's CV error has uncertainty; a simpler model with statistically equivalent performance is preferred as it's less likely to overfit and is easier to interpret.
</details>

### Q10 (4 pts)
What is the condition number in gradient descent? How does it affect convergence?

<details>
<summary><strong>Solution</strong></summary>

The condition number κ = b_max/b_min is the ratio of largest to smallest eigenvalues of the Hessian (for quadratic objectives) or analogous quantities. Large condition number (ill-conditioned problem) means gradient descent converges slowly—the optimal step size is constrained by the largest eigenvalue while progress along directions with small eigenvalues is slow. The convergence rate is approximately (κ-1)/(κ+1).
</details>

### Q11 (4 pts)
Explain how Random Forests reduce overfitting compared to a single decision tree.

<details>
<summary><strong>Solution</strong></summary>

Random Forests use two techniques: (1) Bagging—each tree is trained on a bootstrap sample, introducing diversity through different training subsets. (2) Random feature selection—at each split, only a random subset of features is considered, further decorrelating trees. Averaging predictions from many diverse trees reduces variance while maintaining low bias, leading to better generalization than a single deep tree.
</details>

---

## Section B: Mathematical Derivations (45 points)

### Q12 (10 pts)
Derive the least squares solution for multiple linear regression.

Starting from $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, show that the estimate minimizing RSS is:
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Write the RSS
$$RSS(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

**Step 2**: Expand
$$RSS = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

**Step 3**: Take gradient with respect to β
Using matrix calculus rules ($\nabla_\beta(\mathbf{a}^T\boldsymbol{\beta}) = \mathbf{a}$ and $\nabla_\beta(\boldsymbol{\beta}^T\mathbf{A}\boldsymbol{\beta}) = 2\mathbf{A}\boldsymbol{\beta}$ for symmetric A):

$$\nabla_\beta RSS = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

**Step 4**: Set gradient to zero
$$-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$
$$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

**Step 5**: Solve (assuming X^TX is invertible)
$$\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

**Step 6**: Verify it's a minimum
The Hessian is $\nabla^2 RSS = 2\mathbf{X}^T\mathbf{X}$, which is positive semi-definite (and positive definite if X has full column rank). Therefore, this is indeed a minimum. ∎
</details>

---

### Q13 (10 pts)
Derive the gradient of the binary cross-entropy loss for logistic regression.

For loss $J(\boldsymbol{\beta}) = \sum_{i=1}^n [\log(1+e^{z_i}) - y_i z_i]$ where $z_i = \boldsymbol{\beta}^T\mathbf{x}_i$:

(a) Show that $\frac{\partial J}{\partial z_i} = \sigma(z_i) - y_i$ where $\sigma(z) = \frac{1}{1+e^{-z}}$

(b) Derive $\frac{\partial J}{\partial \beta_j}$

<details>
<summary><strong>Solution</strong></summary>

**(a) Gradient w.r.t. z_i**:

$$\frac{\partial}{\partial z_i}[\log(1+e^{z_i}) - y_i z_i] = \frac{e^{z_i}}{1+e^{z_i}} - y_i$$

Note that:
$$\frac{e^{z}}{1+e^{z}} = \frac{1}{1+e^{-z}} = \sigma(z)$$

Therefore:
$$\boxed{\frac{\partial J}{\partial z_i} = \sigma(z_i) - y_i}$$

**(b) Gradient w.r.t. β_j**:

Using chain rule:
$$\frac{\partial J}{\partial \beta_j} = \sum_{i=1}^n \frac{\partial J}{\partial z_i} \cdot \frac{\partial z_i}{\partial \beta_j}$$

Since $z_i = \sum_k \beta_k x_{ik}$:
$$\frac{\partial z_i}{\partial \beta_j} = x_{ij}$$

Therefore:
$$\boxed{\frac{\partial J}{\partial \beta_j} = \sum_{i=1}^n (\sigma(z_i) - y_i) x_{ij}}$$

In vector form: $\nabla_\beta J = \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$ where $\hat{y}_i = \sigma(z_i)$.
</details>

---

### Q14 (10 pts)
Derive the backpropagation equations for a layer with sigmoid activation.

Given: $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, $\mathbf{u} = \sigma(\mathbf{z})$ where σ is applied element-wise.

Assuming you have $\frac{\partial J}{\partial \mathbf{u}}$, derive:
(a) $\frac{\partial J}{\partial \mathbf{z}}$
(b) $\frac{\partial J}{\partial \mathbf{W}}$
(c) $\frac{\partial J}{\partial \mathbf{x}}$ (for backprop to previous layer)

<details>
<summary><strong>Solution</strong></summary>

**(a) Gradient w.r.t. z**:

Since $u_j = \sigma(z_j)$ for each component:
$$\frac{\partial u_j}{\partial z_j} = \sigma(z_j)(1 - \sigma(z_j)) = u_j(1-u_j)$$

By chain rule (element-wise):
$$\boxed{\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{u}} \odot \mathbf{u} \odot (1-\mathbf{u})}$$

where ⊙ denotes element-wise multiplication.

**(b) Gradient w.r.t. W**:

Since $z_j = \sum_k W_{jk}x_k + b_j$:
$$\frac{\partial z_j}{\partial W_{jk}} = x_k$$

Therefore:
$$\frac{\partial J}{\partial W_{jk}} = \frac{\partial J}{\partial z_j} \cdot x_k$$

In matrix form:
$$\boxed{\frac{\partial J}{\partial \mathbf{W}} = \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^T}$$

**(c) Gradient w.r.t. x (for backprop)**:

$$\frac{\partial z_j}{\partial x_k} = W_{jk}$$

$$\frac{\partial J}{\partial x_k} = \sum_j \frac{\partial J}{\partial z_j} W_{jk}$$

In matrix form:
$$\boxed{\frac{\partial J}{\partial \mathbf{x}} = \mathbf{W}^T \frac{\partial J}{\partial \mathbf{z}}}$$
</details>

---

### Q15 (8 pts)
Prove that for K-means, the optimal cluster center is the mean of the assigned points.

That is, show that $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k}\mathbf{x}_i$ minimizes $\sum_{i \in C_k}\|\mathbf{x}_i - \boldsymbol{\mu}\|^2$.

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Write the objective for cluster k
$$J_k(\boldsymbol{\mu}) = \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}\|^2$$

**Step 2**: Expand the squared norm
$$J_k = \sum_{i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu})^T(\mathbf{x}_i - \boldsymbol{\mu})$$
$$= \sum_{i \in C_k} \left(\mathbf{x}_i^T\mathbf{x}_i - 2\boldsymbol{\mu}^T\mathbf{x}_i + \boldsymbol{\mu}^T\boldsymbol{\mu}\right)$$

**Step 3**: Take gradient w.r.t. μ
$$\nabla_\mu J_k = \sum_{i \in C_k} (-2\mathbf{x}_i + 2\boldsymbol{\mu}) = -2\sum_{i \in C_k}\mathbf{x}_i + 2|C_k|\boldsymbol{\mu}$$

**Step 4**: Set gradient to zero
$$-2\sum_{i \in C_k}\mathbf{x}_i + 2|C_k|\boldsymbol{\mu} = \mathbf{0}$$
$$\boldsymbol{\mu} = \frac{1}{|C_k|}\sum_{i \in C_k}\mathbf{x}_i$$

**Step 5**: Verify it's a minimum
The Hessian is $\nabla^2 J_k = 2|C_k|\mathbf{I}$, which is positive definite. Therefore, this is indeed a minimum. ∎
</details>

---

### Q16 (7 pts)
Prove that PCA reconstruction error equals the sum of squared skipped PC coefficients.

Show: $\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{j=k+1}^d z_j^2$ where $\hat{\mathbf{x}}$ uses the first k PCs.

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Express data in PC basis
Since PCs $\{\mathbf{v}_1,...,\mathbf{v}_d\}$ form an orthonormal basis:
$$\mathbf{x} - \boldsymbol{\mu} = \sum_{j=1}^d z_j\mathbf{v}_j$$

**Step 2**: Write k-PC reconstruction
$$\hat{\mathbf{x}} - \boldsymbol{\mu} = \sum_{j=1}^k z_j\mathbf{v}_j$$

**Step 3**: Compute error
$$\mathbf{x} - \hat{\mathbf{x}} = \sum_{j=1}^d z_j\mathbf{v}_j - \sum_{j=1}^k z_j\mathbf{v}_j = \sum_{j=k+1}^d z_j\mathbf{v}_j$$

**Step 4**: Compute squared norm using orthonormality
$$\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \left\|\sum_{j=k+1}^d z_j\mathbf{v}_j\right\|^2 = \sum_{j=k+1}^d z_j^2 \cdot \|\mathbf{v}_j\|^2 = \sum_{j=k+1}^d z_j^2$$ ∎
</details>

---

## Section C: Computational Problems (45 points)

### Q17 (12 pts)
Given data:

| $x$ | 0 | 1 | 2 | 3 |
|-----|---|---|---|---|
| $y$ | 1 | 2 | 4 | 8 |

(a) Fit a linear model $y = \beta_0 + \beta_1 x$. Find $\hat{\beta}_0$ and $\hat{\beta}_1$. (4 pts)

(b) The data appears exponential. Transform to $\ln(y) = \alpha_0 + \alpha_1 x$ and find the parameters. (4 pts)

(c) Convert back to get $y = ae^{bx}$. What are $a$ and $b$? (2 pts)

(d) Which model (linear or exponential) fits better? How can you tell? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

**(a) Linear fit**:
- $\bar{x} = 1.5$, $\bar{y} = 3.75$
- $\sum(x_i - \bar{x})(y_i - \bar{y}) = (-1.5)(-2.75) + (-0.5)(-1.75) + (0.5)(0.25) + (1.5)(4.25) = 4.125 + 0.875 + 0.125 + 6.375 = 11.5$
- $\sum(x_i - \bar{x})^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5$
- $\hat{\beta}_1 = 11.5/5 = 2.3$
- $\hat{\beta}_0 = 3.75 - 2.3(1.5) = 0.3$

**Answer**: $\hat{y} = 0.3 + 2.3x$

**(b) Transform and fit**:
| $x$ | 0 | 1 | 2 | 3 |
|-----|---|---|---|---|
| $\ln(y)$ | 0 | 0.693 | 1.386 | 2.079 |

- $\bar{x} = 1.5$, $\overline{\ln y} = 1.04$
- $\sum(x_i - \bar{x})(\ln y_i - \overline{\ln y}) = (-1.5)(-1.04) + (-0.5)(-0.347) + (0.5)(0.346) + (1.5)(1.039) = 3.468$
- $\sum(x_i - \bar{x})^2 = 5$
- $\hat{\alpha}_1 = 3.468/5 = 0.694 \approx \ln(2)$
- $\hat{\alpha}_0 = 1.04 - 0.694(1.5) = -0.001 \approx 0$

**Answer**: $\ln(y) = 0 + 0.693x$

**(c) Convert back**:
- $a = e^{\alpha_0} = e^0 = 1$
- $b = \alpha_1 = 0.693 \approx \ln(2)$

**Answer**: $y = e^{0.693x} = 2^x$ (approximately)

**(d) Model comparison**:
The exponential model fits perfectly! Looking at the data: $y = 2^x$ gives exactly 1, 2, 4, 8.

The linear model has visible residuals. We can compute RSS:
- Linear RSS: $(1-0.3)^2 + (2-2.6)^2 + (4-4.9)^2 + (8-7.2)^2 = 0.49 + 0.36 + 0.81 + 0.64 = 2.3$
- Exponential RSS ≈ 0

The exponential model is clearly better.
</details>

---

### Q18 (10 pts)
A neural network has input x, hidden layer with 2 units using ReLU, and linear output.

Weights: $W^H = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$, $b^H = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$, $W^O = [1, 1]$, $b^O = 0$

(a) For input $x = 2$, compute forward pass to get $\hat{y}$ (4 pts)

(b) If true output is $y = 1$ and loss is $J = \frac{1}{2}(y-\hat{y})^2$, compute $\frac{\partial J}{\partial W^H}$ (6 pts)

<details>
<summary><strong>Solution</strong></summary>

**(a) Forward pass**:

Hidden layer:
$$\mathbf{z}^H = W^H x + b^H = \begin{pmatrix} 1 \\ -1 \end{pmatrix}(2) + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ -2 \end{pmatrix}$$

$$\mathbf{u}^H = \text{ReLU}(\mathbf{z}^H) = \begin{pmatrix} \max(0,2) \\ \max(0,-2) \end{pmatrix} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$$

Output layer:
$$z^O = W^O \mathbf{u}^H + b^O = [1,1]\begin{pmatrix} 2 \\ 0 \end{pmatrix} + 0 = 2$$

$$\hat{y} = z^O = 2$$ (linear)

**Answer**: $\hat{y} = 2$

**(b) Backpropagation**:

Loss gradient:
$$\frac{\partial J}{\partial \hat{y}} = -(y - \hat{y}) = -(1 - 2) = 1$$

Since output is linear, $\frac{\partial J}{\partial z^O} = 1$

Gradient to hidden activations:
$$\frac{\partial J}{\partial \mathbf{u}^H} = (W^O)^T \cdot \frac{\partial J}{\partial z^O} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \cdot 1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

ReLU derivative:
$$\frac{\partial \mathbf{u}^H}{\partial \mathbf{z}^H} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$ (since $z_1 > 0$, $z_2 < 0$)

Gradient to hidden pre-activations:
$$\frac{\partial J}{\partial \mathbf{z}^H} = \frac{\partial J}{\partial \mathbf{u}^H} \odot \mathbf{1}_{z>0} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \odot \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

Gradient to hidden weights:
$$\frac{\partial J}{\partial W^H} = \frac{\partial J}{\partial \mathbf{z}^H} \cdot x^T = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \cdot 2 = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$$

**Answer**: $\frac{\partial J}{\partial W^H} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$
</details>

---

### Q19 (8 pts)
For a CNN layer: input shape (8, 8, 3), kernel shape (3, 3, 3, 8), valid convolution.

(a) What is the output shape? (2 pts)
(b) How many parameters (including bias)? (2 pts)
(c) After 2×2 max pooling with stride 2, what is the output shape? (2 pts)
(d) How many multiplications for the convolution? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

**(a) Output shape**:
- Height: 8 - 3 + 1 = 6
- Width: 8 - 3 + 1 = 6
- Channels: 8

**Answer**: (6, 6, 8)

**(b) Parameters**:
- Weights: 3 × 3 × 3 × 8 = 216
- Biases: 8

**Answer**: 216 + 8 = **224 parameters**

**(c) After pooling**:
- Height: 6/2 = 3
- Width: 6/2 = 3
- Channels: 8

**Answer**: (3, 3, 8)

**(d) Multiplications**:
- Per output pixel: 3 × 3 × 3 = 27
- Number of output pixels: 6 × 6 × 8 = 288

**Answer**: 27 × 288 = **7,776 multiplications**
</details>

---

### Q20 (8 pts)
Given data matrix X (already centered):
$$\tilde{X} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \\ -2 & 0 \\ 0 & -1 \end{pmatrix}$$

(a) Compute the covariance matrix Q (3 pts)
(b) Find the eigenvalues and eigenvectors (3 pts)
(c) What percentage of variance is explained by the first PC? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

**(a) Covariance matrix**:
$$Q = \frac{1}{n-1}\tilde{X}^T\tilde{X} = \frac{1}{3}\begin{pmatrix} 2 & 0 & -2 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}\begin{pmatrix} 2 & 0 \\ 0 & 1 \\ -2 & 0 \\ 0 & -1 \end{pmatrix}$$

$$= \frac{1}{3}\begin{pmatrix} 8 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 8/3 & 0 \\ 0 & 2/3 \end{pmatrix}$$

**(b) Eigendecomposition**:
Since Q is diagonal, eigenvalues are the diagonal elements:
- $\lambda_1 = 8/3 \approx 2.67$
- $\lambda_2 = 2/3 \approx 0.67$

Eigenvectors:
- $\mathbf{v}_1 = [1, 0]^T$
- $\mathbf{v}_2 = [0, 1]^T$

**(c) Variance explained by first PC**:
$$PVE_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{8/3}{8/3 + 2/3} = \frac{8/3}{10/3} = \frac{8}{10} = 0.8$$

**Answer**: **80%**
</details>

---

### Q21 (7 pts)
Five points: (0,0), (1,1), (2,0), (5,5), (6,5)

Run one iteration of K-means with K=2, starting with centers $\mu_1 = (0,0)$, $\mu_2 = (6,5)$.

(a) Assign each point to its nearest center (3 pts)
(b) Compute new centers (2 pts)
(c) What is the K-means objective J? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

**(a) Assignments**:

| Point | Dist to (0,0) | Dist to (6,5) | Assign |
|-------|---------------|---------------|--------|
| (0,0) | 0 | √61 ≈ 7.81 | C1 |
| (1,1) | √2 ≈ 1.41 | √41 ≈ 6.40 | C1 |
| (2,0) | 2 | √41 ≈ 6.40 | C1 |
| (5,5) | √50 ≈ 7.07 | 1 | C2 |
| (6,5) | √61 ≈ 7.81 | 0 | C2 |

Cluster 1: {(0,0), (1,1), (2,0)}
Cluster 2: {(5,5), (6,5)}

**(b) New centers**:
$$\mu_1^{new} = \frac{1}{3}[(0,0) + (1,1) + (2,0)] = (1, 1/3) \approx (1, 0.33)$$
$$\mu_2^{new} = \frac{1}{2}[(5,5) + (6,5)] = (5.5, 5)$$

**(c) K-means objective**:
For cluster 1 (using original centers for this iteration):
- $\|(0,0)-(0,0)\|^2 = 0$
- $\|(1,1)-(0,0)\|^2 = 2$
- $\|(2,0)-(0,0)\|^2 = 4$

Sum for C1 = 6

For cluster 2:
- $\|(5,5)-(6,5)\|^2 = 1$
- $\|(6,5)-(6,5)\|^2 = 0$

Sum for C2 = 1

**Answer**: J = 6 + 1 = **7**
</details>

---

## Section D: Pseudocode (20 points)

*Write pseudocode (not specific implementation) for the following algorithms.*

### Q22 (6 pts)
Write pseudocode for K-fold cross-validation to select the best regularization parameter λ for Ridge regression.

<details>
<summary><strong>Solution</strong></summary>

```
Input: X, y, list of lambda values, K (number of folds)

# Split data into K folds
folds = split_into_k_folds(X, y, K)

# For each lambda value
for each lambda in lambda_values:
    cv_errors = []
    
    for k = 1 to K:
        # Create train/test split
        X_train, y_train = combine folds except k
        X_test, y_test = fold k
        
        # Fit Ridge regression
        beta = (X_train^T * X_train + lambda * I)^(-1) * X_train^T * y_train
        
        # Compute test error
        y_pred = X_test * beta
        error = mean((y_test - y_pred)^2)
        cv_errors.append(error)
    
    mean_cv_error[lambda] = mean(cv_errors)
    std_cv_error[lambda] = std(cv_errors)

# Select best lambda
best_lambda = lambda with minimum mean_cv_error

# Optional: one standard error rule
threshold = min(mean_cv_error) + std_cv_error[argmin(mean_cv_error)] / sqrt(K)
best_lambda_1se = largest lambda where mean_cv_error <= threshold

return best_lambda (or best_lambda_1se)
```
</details>

---

### Q23 (7 pts)
Write pseudocode for training a single-hidden-layer neural network with ReLU activation using gradient descent.

<details>
<summary><strong>Solution</strong></summary>

```
Input: X (n x d), y (n x 1), hidden_size, learning_rate, num_epochs

# Initialize weights randomly
W1 = random_normal(d, hidden_size)
b1 = zeros(hidden_size)
W2 = random_normal(hidden_size, 1)
b2 = 0

for epoch = 1 to num_epochs:
    # Forward pass
    Z1 = X @ W1 + b1          # (n x hidden_size)
    U1 = max(0, Z1)            # ReLU activation
    Z2 = U1 @ W2 + b2          # (n x 1)
    y_hat = Z2                  # Linear output
    
    # Compute loss
    loss = mean((y - y_hat)^2)
    
    # Backward pass
    # Output layer gradients
    dL_dZ2 = -2/n * (y - y_hat)     # (n x 1)
    dL_dW2 = U1^T @ dL_dZ2          # (hidden x 1)
    dL_db2 = sum(dL_dZ2)            # scalar
    
    # Hidden layer gradients
    dL_dU1 = dL_dZ2 @ W2^T          # (n x hidden)
    dL_dZ1 = dL_dU1 * (Z1 > 0)      # ReLU derivative
    dL_dW1 = X^T @ dL_dZ1           # (d x hidden)
    dL_db1 = sum(dL_dZ1, axis=0)    # (hidden,)
    
    # Update weights
    W2 = W2 - learning_rate * dL_dW2
    b2 = b2 - learning_rate * dL_db2
    W1 = W1 - learning_rate * dL_dW1
    b1 = b1 - learning_rate * dL_db1

return W1, b1, W2, b2
```
</details>

---

### Q24 (7 pts)
Write pseudocode for building a decision tree for classification using Gini impurity.

<details>
<summary><strong>Solution</strong></summary>

```
function build_tree(X, y, max_depth):
    # Base cases
    if all labels in y are the same:
        return Leaf(majority_class(y))
    if max_depth == 0 or num_samples < min_samples:
        return Leaf(majority_class(y))
    
    # Find best split
    best_gain = 0
    best_feature = None
    best_threshold = None
    
    current_gini = compute_gini(y)
    
    for each feature j:
        for each threshold t in unique values of X[:, j]:
            # Split data
            left_mask = X[:, j] <= t
            right_mask = X[:, j] > t
            
            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue
            
            # Compute weighted Gini after split
            n_left = sum(left_mask)
            n_right = sum(right_mask)
            n_total = n_left + n_right
            
            gini_left = compute_gini(y[left_mask])
            gini_right = compute_gini(y[right_mask])
            weighted_gini = (n_left/n_total)*gini_left + (n_right/n_total)*gini_right
            
            gain = current_gini - weighted_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_threshold = t
    
    # If no good split found
    if best_gain == 0:
        return Leaf(majority_class(y))
    
    # Recursively build children
    left_mask = X[:, best_feature] <= best_threshold
    left_child = build_tree(X[left_mask], y[left_mask], max_depth - 1)
    right_child = build_tree(X[~left_mask], y[~left_mask], max_depth - 1)
    
    return Node(feature=best_feature, threshold=best_threshold,
                left=left_child, right=right_child)

function compute_gini(y):
    n = len(y)
    if n == 0:
        return 0
    p = count(y == 1) / n
    return 2 * p * (1 - p)
```
</details>

---

## Answer Summary

### Key Numerical Answers:

| Question | Answer |
|----------|--------|
| Q17(a) | β₀ = 0.3, β₁ = 2.3 |
| Q17(c) | a = 1, b = 0.693 |
| Q18(a) | ŷ = 2 |
| Q18(b) | ∂J/∂W^H = [2, 0]ᵀ |
| Q19(a) | (6, 6, 8) |
| Q19(b) | 224 parameters |
| Q19(d) | 7,776 multiplications |
| Q20(c) | 80% |
| Q21(c) | J = 7 |

### Key Formulas to Remember:

1. **Least Squares**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

2. **Logistic Gradient**: $\nabla J = \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$

3. **Backprop through sigmoid**: $\frac{\partial J}{\partial \mathbf{z}} = \frac{\partial J}{\partial \mathbf{u}} \odot \mathbf{u} \odot (1-\mathbf{u})$

4. **PCA reconstruction error**: $\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{j=k+1}^d z_j^2$

5. **K-means center**: $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k}\mathbf{x}_i$

