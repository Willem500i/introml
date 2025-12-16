# Worked Examples - Step by Step Solutions

This document walks you through representative problems from each unit, explaining the thought process and solution methodology.

---

## Unit 2: Simple Linear Regression

### Example 1: Computing Least Squares Parameters

**Problem**: Given data samples $(x_i, y_i)$:

| $x_i$ | 0 | 1 | 2 | 3 | 4 |
|-------|---|---|---|---|---|
| $y_i$ | 0 | 2 | 3 | 8 | 17 |

Find the least squares parameters for $y = \beta_0 + \beta_1 x$.

**Solution Walkthrough**:

**Step 1**: Compute sample means
$$\bar{x} = \frac{0+1+2+3+4}{5} = \frac{10}{5} = 2$$
$$\bar{y} = \frac{0+2+3+8+17}{5} = \frac{30}{5} = 6$$

**Step 2**: Compute deviations and products

| $i$ | $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i-\bar{x})(y_i-\bar{y})$ | $(x_i-\bar{x})^2$ |
|-----|-------|-------|-----------------|-----------------|------------------------------|-------------------|
| 1 | 0 | 0 | -2 | -6 | 12 | 4 |
| 2 | 1 | 2 | -1 | -4 | 4 | 1 |
| 3 | 2 | 3 | 0 | -3 | 0 | 0 |
| 4 | 3 | 8 | 1 | 2 | 2 | 1 |
| 5 | 4 | 17 | 2 | 11 | 22 | 4 |
| **Sum** | | | | | **40** | **10** |

**Step 3**: Compute $\beta_1$ (slope)
$$\hat{\beta}_1 = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2} = \frac{40}{10} = 4$$

**Step 4**: Compute $\beta_0$ (intercept)
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} = 6 - 4(2) = 6 - 8 = -2$$

**Answer**: $\hat{y} = -2 + 4x$

---

### Example 2: Linearizing an Exponential Model

**Problem**: A chemical concentration follows $z(t) = z_0 e^{-\alpha t}$. How can we use linear regression to estimate $z_0$ and $\alpha$?

**Solution Walkthrough**:

**Step 1**: Take logarithm of both sides
$$\ln(z) = \ln(z_0 e^{-\alpha t}) = \ln(z_0) + \ln(e^{-\alpha t}) = \ln(z_0) - \alpha t$$

**Step 2**: Define new variables
- Let $y = \ln(z)$
- Let $\beta_0 = \ln(z_0)$
- Let $\beta_1 = -\alpha$

**Step 3**: The model becomes linear
$$y = \beta_0 + \beta_1 t$$

**Step 4**: After fitting, recover original parameters
- $z_0 = e^{\hat{\beta}_0}$
- $\alpha = -\hat{\beta}_1$

**Python pseudocode**:
```python
y = np.log(z)  # Transform response
beta1 = np.sum((t - t.mean()) * (y - y.mean())) / np.sum((t - t.mean())**2)
beta0 = y.mean() - beta1 * t.mean()
z0 = np.exp(beta0)
alpha = -beta1
```

---

## Unit 3: Multiple Linear Regression

### Example 3: Fitting with Multiple Features

**Problem**: Given data with two features:

| $x_{i1}$ | 0 | 0 | 1 | 1 |
|----------|---|---|---|---|
| $x_{i2}$ | 0 | 1 | 0 | 1 |
| $y_i$ | 1 | 4 | 3 | 7 |

Find $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$.

**Solution Walkthrough**:

**Step 1**: Set up the design matrix and response vector
$$\mathbf{X} = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 4 \\ 3 \\ 7 \end{pmatrix}$$

**Step 2**: Compute $\mathbf{X}^T\mathbf{X}$
$$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 4 & 2 & 2 \\ 2 & 2 & 1 \\ 2 & 1 & 2 \end{pmatrix}$$

**Step 3**: Compute $\mathbf{X}^T\mathbf{y}$
$$\mathbf{X}^T\mathbf{y} = \begin{pmatrix} 1+4+3+7 \\ 0+0+3+7 \\ 0+4+0+7 \end{pmatrix} = \begin{pmatrix} 15 \\ 10 \\ 11 \end{pmatrix}$$

**Step 4**: Solve the normal equations
From $\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$:
$$4\beta_0 + 2\beta_1 + 2\beta_2 = 15$$
$$2\beta_0 + 2\beta_1 + 1\beta_2 = 10$$
$$2\beta_0 + 1\beta_1 + 2\beta_2 = 11$$

Solving (by elimination or matrix inverse):
- $\beta_0 = 0.75$
- $\beta_1 = 2.5$
- $\beta_2 = 3.5$

**Answer**: $\hat{y} = 0.75 + 2.5x_1 + 3.5x_2$

---

### Example 4: Transforming to Linear Form

**Problem**: Write $\hat{y} = (a_1 x_1 + a_2 x_2)e^{-x_1-x_2}$ as $\hat{y} = \boldsymbol{\beta}^T\boldsymbol{\phi}(\mathbf{x})$.

**Solution Walkthrough**:

**Step 1**: Expand and identify linear structure
$$\hat{y} = a_1 x_1 e^{-x_1-x_2} + a_2 x_2 e^{-x_1-x_2}$$

**Step 2**: Define basis functions
$$\phi_1(\mathbf{x}) = x_1 e^{-x_1-x_2}$$
$$\phi_2(\mathbf{x}) = x_2 e^{-x_1-x_2}$$

**Step 3**: Define parameter vector
$$\boldsymbol{\beta} = \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}$$

**Step 4**: Verify

$$\hat{y} = \beta_1 \phi_1 + \beta_2 \phi_2 = a_1 \cdot x_1 e^{-x_1-x_2} + a_2 \cdot x_2 e^{-x_1-x_2}$$

This matches the original form. ✓

**Recovery**: $a_1 = \beta_1$, $a_2 = \beta_2$

---

## Unit 4: Model Selection

### Example 5: Bias Calculation with Undermodeling

**Problem**: True function is $f_0(x) = \beta_{00} + \beta_{01}x + \beta_{02}x^2$, but we fit $\hat{y} = \beta_0 + \beta_1 x$. What is the bias?

**Solution Walkthrough**:

**Step 1**: Express training data using true model
$$y_i = \beta_{00} + \beta_{01}x_i + \beta_{02}x_i^2$$

**Step 2**: Write the least squares estimate
$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

where $\mathbf{X}$ is the design matrix for the linear model.

**Step 3**: Substitute $\mathbf{y}$
$$\mathbf{y} = \beta_{00}\mathbf{1} + \beta_{01}\mathbf{x} + \beta_{02}\mathbf{x}^{(2)}$$

where $\mathbf{x}^{(2)} = [x_1^2, x_2^2, \ldots, x_n^2]^T$.

**Step 4**: The bias at test point $x$ is
$$\text{Bias}(x) = \mathbb{E}[f(x,\hat{\boldsymbol{\beta}})] - f_0(x)$$

Since the model doesn't include $x^2$, the linear fit will have systematic error.

**Key insight**: The bias depends on the distribution of training $x_i$ values and how well a line can approximate the quadratic over that range.

---

## Unit 5: LASSO

### Example 6: Soft Thresholding Derivation

**Problem**: Find $\hat{w} = \arg\min_w \frac{1}{2}(y-w)^2 + \lambda|w|$

**Solution Walkthrough**:

**Case 1: $w > 0$**
In this region, $|w| = w$, so:
$$J(w) = \frac{1}{2}(y-w)^2 + \lambda w$$
$$J'(w) = -(y-w) + \lambda = w - y + \lambda$$

Setting $J'(w) = 0$: $w = y - \lambda$

This is valid only if $y - \lambda > 0$, i.e., $y > \lambda$.

**Case 2: $w < 0$**
Here $|w| = -w$:
$$J(w) = \frac{1}{2}(y-w)^2 - \lambda w$$
$$J'(w) = -(y-w) - \lambda = w - y - \lambda$$

Setting $J'(w) = 0$: $w = y + \lambda$

Valid only if $y + \lambda < 0$, i.e., $y < -\lambda$.

**Case 3: $|y| \leq \lambda$**
Neither case above applies, so minimum is at $w = 0$.

**Final Answer (Soft Thresholding)**:
$$\hat{w} = \begin{cases} y - \lambda & y > \lambda \\ 0 & |y| \leq \lambda \\ y + \lambda & y < -\lambda \end{cases}$$

---

## Unit 6: Logistic Regression

### Example 7: Decision Boundaries

**Problem**: Given $P(y=1|\mathbf{x}) = \frac{1}{1+e^{-z}}$ with $z = 1 + 2x_1 + 3x_2$, find:
(a) Where is $P(y=1) > P(y=0)$?
(b) Where is $P(y=1) > 0.8$?

**Solution Walkthrough**:

**(a) Decision boundary**

$P(y=1) > P(y=0)$ means $P(y=1) > 0.5$

Since $\sigma(z) > 0.5$ iff $z > 0$:
$$1 + 2x_1 + 3x_2 > 0$$
$$x_2 > -\frac{1}{3} - \frac{2}{3}x_1$$

**Answer**: The half-plane above the line $x_2 = -\frac{1}{3} - \frac{2}{3}x_1$

**(b) High confidence region**

$P(y=1) > 0.8$ means $\sigma(z) > 0.8$

Solve: $\frac{1}{1+e^{-z}} > 0.8$
$$1 > 0.8(1+e^{-z})$$
$$0.2 > 0.8 e^{-z}$$
$$e^{-z} < 0.25$$
$$-z < \ln(0.25) = -\ln(4) \approx -1.386$$
$$z > 1.386$$

So: $1 + 2x_1 + 3x_2 > 1.386$
$$2x_1 + 3x_2 > 0.386$$

**Answer**: The half-plane above $x_2 = 0.129 - \frac{2}{3}x_1$

---

### Example 8: Gradient of Cross-Entropy Loss

**Problem**: Compute $\frac{\partial J}{\partial \beta_j}$ for $J = \sum_i [\ln(1+e^{z_i}) - y_i z_i]$

**Solution Walkthrough**:

**Step 1**: Identify the chain
$$J \to z_i \to \beta_j$$

where $z_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots$

**Step 2**: Compute $\frac{\partial J}{\partial z_i}$
$$\frac{\partial}{\partial z_i}[\ln(1+e^{z_i}) - y_i z_i] = \frac{e^{z_i}}{1+e^{z_i}} - y_i = \sigma(z_i) - y_i$$

**Step 3**: Compute 
$\frac{\partial z_i}{\partial \beta_j} = x_{ij}$ (or 1 if j = 0)

**Step 4**: Apply chain rule
$$\frac{\partial J}{\partial \beta_j} = \sum_i \frac{\partial J}{\partial z_i} \cdot \frac{\partial z_i}{\partial \beta_j} = \sum_i (\sigma(z_i) - y_i) x_{ij}$$

**In matrix form**: $\nabla_\beta J = \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$

---

## Unit 7: Optimization

### Example 9: Gradient Descent Convergence

**Problem**: For $J(\mathbf{w}) = \frac{1}{2}b_1 w_1^2 + \frac{1}{2}b_2 w_2^2$ with $b_2 > b_1 > 0$, find step size for convergence.

**Solution Walkthrough**:

**Step 1**: Compute gradient
$$\nabla J = \begin{pmatrix} b_1 w_1 \\ b_2 w_2 \end{pmatrix}$$

**Step 2**: Write update rule
$$w_1^{(k+1)} = w_1^{(k)} - \alpha b_1 w_1^{(k)} = (1 - \alpha b_1) w_1^{(k)}$$
$$w_2^{(k+1)} = w_2^{(k)} - \alpha b_2 w_2^{(k)} = (1 - \alpha b_2) w_2^{(k)}$$

**Step 3**: Convergence condition
For $w_i^{(k)} \to 0$, we need $|\rho_i| < 1$ where $\rho_i = 1 - \alpha b_i$

For $\rho_1 = 1 - \alpha b_1$:
$$-1 < 1 - \alpha b_1 < 1$$
$$0 < \alpha b_1 < 2$$
$$0 < \alpha < \frac{2}{b_1}$$

For $\rho_2 = 1 - \alpha b_2$:
$$0 < \alpha < \frac{2}{b_2}$$

**Step 4**: Combined condition
Since $b_2 > b_1$, we have $\frac{2}{b_2} < \frac{2}{b_1}$

**Answer**: Convergence requires $0 < \alpha < \frac{2}{b_2}$

---

## Unit 8: SVM

### Example 10: Computing Margin

**Problem**: Given classifier $\hat{y} = \text{sign}(b + w_1 x_1 + w_2 x_2)$ with $b=-0.5$, $w_1=0$, $w_2=1$, find margin for:

| $x_{i1}$ | 0 | 1 | 1 | 2 |
|----------|---|---|---|---|
| $x_{i2}$ | 0 | 0.3 | 0.7 | 1 |
| $y_i$ | -1 | -1 | 1 | 1 |

**Solution Walkthrough**:

**Step 1**: Compute $z_i = b + w_1 x_{i1} + w_2 x_{i2}$ for each point
- $z_1 = -0.5 + 0(0) + 1(0) = -0.5$
- $z_2 = -0.5 + 0(1) + 1(0.3) = -0.2$
- $z_3 = -0.5 + 0(1) + 1(0.7) = 0.2$
- $z_4 = -0.5 + 0(2) + 1(1) = 0.5$

**Step 2**: Compute $y_i z_i$ (functional margin)
- $y_1 z_1 = (-1)(-0.5) = 0.5$
- $y_2 z_2 = (-1)(-0.2) = 0.2$
- $y_3 z_3 = (1)(0.2) = 0.2$
- $y_4 z_4 = (1)(0.5) = 0.5$

**Step 3**: Find $\gamma = \min_i y_i z_i = 0.2$

**Step 4**: Compute $\|\mathbf{w}\| = \sqrt{0^2 + 1^2} = 1$

**Step 5**: Geometric margin
$$m = \frac{\gamma}{\|\mathbf{w}\|} = \frac{0.2}{1} = 0.2$$

**Support vectors**: Points 2 and 3 (they achieve minimum $y_i z_i$)

---

## Unit 9: Neural Networks

### Example 11: Forward Propagation

**Problem**: Given weights:
$$W^H = \begin{pmatrix} -1 \\ 1 \\ 1 \end{pmatrix}, \quad b^H = \begin{pmatrix} -1 \\ 1 \\ -2 \end{pmatrix}$$

with ReLU activation, compute hidden outputs for $x = 0$.

**Solution Walkthrough**:

**Step 1**: Compute pre-activations
$$z_1^H = w_1^H \cdot x + b_1^H = (-1)(0) + (-1) = -1$$
$$z_2^H = w_2^H \cdot x + b_2^H = (1)(0) + (1) = 1$$
$$z_3^H = w_3^H \cdot x + b_3^H = (1)(0) + (-2) = -2$$

**Step 2**: Apply ReLU $u = \max(0, z)$
$$u_1^H = \max(0, -1) = 0$$
$$u_2^H = \max(0, 1) = 1$$
$$u_3^H = \max(0, -2) = 0$$

**Answer**: $\mathbf{u}^H = [0, 1, 0]^T$

---

### Example 12: Backpropagation

**Problem**: Given $\frac{\partial J}{\partial \hat{y}} = -2(y - \hat{y})$ and sigmoid output, find $\frac{\partial J}{\partial z^{out}}$.

**Solution Walkthrough**:

**Step 1**: Recall sigmoid derivative
$$\frac{\partial \hat{y}}{\partial z^{out}} = \hat{y}(1-\hat{y})$$

**Step 2**: Apply chain rule
$$\frac{\partial J}{\partial z^{out}} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{out}}$$
$$= -2(y - \hat{y}) \cdot \hat{y}(1-\hat{y})$$

**For hidden layer with weights $\mathbf{W}^{out}$**:
$$\frac{\partial J}{\partial \mathbf{u}^H} = (\mathbf{W}^{out})^T \frac{\partial J}{\partial z^{out}}$$

---

## Unit 10: CNNs

### Example 13: 2D Convolution

**Problem**: Compute $Z[i,j] = \sum_{k_1,k_2} W[k_1,k_2]X[i+k_1,j+k_2]$ for:
$$X = \begin{pmatrix} 0 & 0 & 0 \\ 0 & 3 & 3 \\ 0 & 3 & 2 \end{pmatrix}, \quad W = \begin{pmatrix} 1 & -1 \\ 1 & -1 \end{pmatrix}$$

**Solution Walkthrough**:

**Step 1**: Determine output size
- Input: $3 \times 3$
- Kernel: $2 \times 2$
- Output: $(3-2+1) \times (3-2+1) = 2 \times 2$

**Step 2**: Compute each output pixel

$Z[0,0]$: Window covers $X[0:2, 0:2]$
$$Z[0,0] = 1(0) + (-1)(0) + 1(0) + (-1)(3) = -3$$

$Z[0,1]$: Window covers $X[0:2, 1:3]$
$$Z[0,1] = 1(0) + (-1)(0) + 1(3) + (-1)(3) = 0$$

$Z[1,0]$: Window covers $X[1:3, 0:2]$
$$Z[1,0] = 1(0) + (-1)(3) + 1(0) + (-1)(3) = -6$$

$Z[1,1]$: Window covers $X[1:3, 1:3]$
$$Z[1,1] = 1(3) + (-1)(3) + 1(3) + (-1)(2) = 1$$

**Answer**: $Z = \begin{pmatrix} -3 & 0 \\ -6 & 1 \end{pmatrix}$

---

## Unit 11: PCA

### Example 14: PCA Reconstruction

**Problem**: Given mean $\boldsymbol{\mu} = [1, 0, 2]$ and first two PCs:
$$\mathbf{v}_1 = \frac{1}{\sqrt{2}}[1, 1, 0], \quad \mathbf{v}_2 = \frac{1}{\sqrt{2}}[1, -1, 0]$$

Find PC coefficients and reconstruction of $\mathbf{x} = [2, 3, 4]$.

**Solution Walkthrough**:

**Step 1**: Center the data
$$\tilde{\mathbf{x}} = \mathbf{x} - \boldsymbol{\mu} = [2-1, 3-0, 4-2] = [1, 3, 2]$$

**Step 2**: Compute PC coefficients
$$z_1 = \mathbf{v}_1^T \tilde{\mathbf{x}} = \frac{1}{\sqrt{2}}(1 \cdot 1 + 1 \cdot 3 + 0 \cdot 2) = \frac{4}{\sqrt{2}} = 2\sqrt{2}$$
$$z_2 = \mathbf{v}_2^T \tilde{\mathbf{x}} = \frac{1}{\sqrt{2}}(1 \cdot 1 + (-1) \cdot 3 + 0 \cdot 2) = \frac{-2}{\sqrt{2}} = -\sqrt{2}$$

**Step 3**: Reconstruct using 2 PCs
$$\hat{\mathbf{x}} = \boldsymbol{\mu} + z_1 \mathbf{v}_1 + z_2 \mathbf{v}_2$$
$$= [1, 0, 2] + 2\sqrt{2} \cdot \frac{1}{\sqrt{2}}[1, 1, 0] + (-\sqrt{2}) \cdot \frac{1}{\sqrt{2}}[1, -1, 0]$$
$$= [1, 0, 2] + [2, 2, 0] + [-1, 1, 0]$$
$$= [2, 3, 2]$$

**Step 4**: Reconstruction error
$$\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \|[2,3,4] - [2,3,2]\|^2 = \|[0,0,2]\|^2 = 4$$

This equals the sum of squared coefficients for skipped PCs (i.e., $z_3^2 = 4$).

---

## Unit 12: Clustering

### Example 15: K-Means Iteration

**Problem**: Given points $(0,0), (1,0), (0,1), (2,2), (2,3)$ and initial centers $\boldsymbol{\mu}_1 = (0,0)$, $\boldsymbol{\mu}_2 = (1,0)$, perform one K-means iteration.

**Solution Walkthrough**:

**Step 1**: Compute distances to each center

| Point | Dist to $(0,0)$ | Dist to $(1,0)$ | Assign |
|-------|-----------------|-----------------|--------|
| $(0,0)$ | 0 | 1 | 1 |
| $(1,0)$ | 1 | 0 | 2 |
| $(0,1)$ | 1 | $\sqrt{2} \approx 1.41$ | 1 |
| $(2,2)$ | $\sqrt{8} \approx 2.83$ | $\sqrt{5} \approx 2.24$ | 2 |
| $(2,3)$ | $\sqrt{13} \approx 3.61$ | $\sqrt{10} \approx 3.16$ | 2 |

**Step 2**: Update cluster assignments
- Cluster 1: $\{(0,0), (0,1)\}$
- Cluster 2: $\{(1,0), (2,2), (2,3)\}$

**Step 3**: Compute new centers
$$\boldsymbol{\mu}_1^{new} = \frac{1}{2}[(0,0) + (0,1)] = (0, 0.5)$$
$$\boldsymbol{\mu}_2^{new} = \frac{1}{3}[(1,0) + (2,2) + (2,3)] = \left(\frac{5}{3}, \frac{5}{3}\right) \approx (1.67, 1.67)$$

**Answer**: New centers are $(0, 0.5)$ and $(1.67, 1.67)$

---

## Unit 13: Decision Trees

### Example 16: Finding Split Points

**Problem**: For classification with features $x_1, x_2$ and binary labels, how do you find the best split?

**Solution Walkthrough**:

**Step 1**: For each feature $x_j$:
- Sort samples by $x_j$
- Consider splits at midpoints between consecutive values

**Step 2**: For each potential split:
- Compute impurity of resulting child nodes
- Weighted average by number of samples

**Step 3**: Choose split with largest impurity reduction

**Example calculation with Gini**:
- Parent has 6 samples: 4 positive, 2 negative
- Gini(parent) = $2 \cdot \frac{4}{6} \cdot \frac{2}{6} = \frac{8}{18} \approx 0.444$

After split:
- Left child: 3 positive, 0 negative → Gini = 0
- Right child: 1 positive, 2 negative → Gini = $2 \cdot \frac{1}{3} \cdot \frac{2}{3} = \frac{4}{9} \approx 0.444$

Weighted average: $\frac{3}{6}(0) + \frac{3}{6}(0.444) = 0.222$

**Information gain**: $0.444 - 0.222 = 0.222$

