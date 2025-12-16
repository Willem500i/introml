# Practice Test 2: Optimization, SVMs, and Neural Networks (Units 7-10)

**Time Estimate**: 90 minutes
**Total Points**: 100

---

## Part A: Conceptual Questions (30 points)

### Question 1 (6 points)
Explain the following concepts in gradient descent:
(a) What is the step size (learning rate) and what happens if it's too large or too small? (3 pts)
(b) What is a local minimum vs global minimum, and when can we guarantee finding the global minimum? (3 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) The **step size** (α) determines how large each update step is: $w^{(k+1)} = w^{(k)} - \alpha \nabla J(w^{(k)})$
- **Too large**: May overshoot the minimum, causing oscillation or divergence
- **Too small**: Very slow convergence, may take too many iterations to reach minimum

(b) 
- **Local minimum**: Point where function value is lower than all nearby points
- **Global minimum**: Point with the lowest function value over entire domain
- **Guarantee global**: When the function is **convex**, every local minimum is a global minimum. Linear regression, logistic regression (for separable data), and ridge/LASSO all have convex objectives.
</details>

---

### Question 2 (8 points)
For Support Vector Machines:
(a) What is the margin and why do we want to maximize it? (2 pts)
(b) What is the difference between hard-margin and soft-margin SVM? (2 pts)
(c) What is a support vector? (2 pts)
(d) How does the kernel trick allow SVMs to learn nonlinear boundaries? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) The **margin** is the distance from the decision boundary to the nearest training points. Maximizing it provides better generalization—the classifier is most "confident" about points near the boundary.

(b) 
- **Hard-margin SVM**: Requires all points to be correctly classified with margin ≥ 1. Only works if data is linearly separable.
- **Soft-margin SVM**: Allows some points to violate the margin (with slack variables εᵢ). Controlled by parameter C that trades off margin size vs violations.

(c) **Support vectors** are the training points that lie on the margin boundaries (εᵢ = 0) or violate them (εᵢ > 0). They are the only points that affect the decision boundary—removing other points doesn't change the classifier.

(d) The **kernel trick** computes inner products in a high-dimensional feature space without explicitly computing the transformation. Using kernel K(x,x'), we can implicitly work in spaces where linear separation becomes possible, even though the boundary is nonlinear in original space. Example: RBF kernel K(x,x') = exp(-γ||x-x'||²).
</details>

---

### Question 3 (8 points)
For neural networks:
(a) What is the purpose of the activation function? What happens if we use only linear activations? (3 pts)
(b) What is backpropagation and why is it computationally efficient? (3 pts)
(c) What is the ReLU activation? What are its advantages over sigmoid? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Activation functions** introduce nonlinearity, allowing networks to learn complex patterns. If we use only linear activations, the entire network collapses to a single linear transformation: $W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + W_2 b_1 + b_2$ is still linear. We'd just have a fancy linear model.

(b) **Backpropagation** is an algorithm for computing gradients of the loss with respect to all network parameters. It's efficient because:
- Uses chain rule to decompose gradients layer by layer
- Reuses intermediate computations (forward pass values, previous gradients)
- Complexity is O(number of weights) per sample, not exponential

(c) **ReLU**: g(z) = max(0, z)
**Advantages over sigmoid**:
- No vanishing gradient problem (gradient is 1 for z > 0)
- Faster computation (no exponentials)
- Sparse activations (many neurons output 0)
- Empirically trains faster and often performs better
</details>

---

### Question 4 (8 points)
For CNNs:
(a) Why are convolutional layers preferred over fully-connected layers for images? (3 pts)
(b) What is the receptive field of a neuron in a CNN? (2 pts)
(c) What is the purpose of pooling layers? (3 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) Convolutional layers are preferred because:
- **Parameter sharing**: Same filter applied across all spatial locations → far fewer parameters than FC
- **Translation invariance**: Learned features work regardless of position
- **Local connectivity**: Exploits spatial structure—nearby pixels are more related
- **Scalability**: FC layers on images would have billions of parameters

(b) The **receptive field** of a neuron is the region of the original input that can influence that neuron's output. It grows with network depth—deeper layers have larger receptive fields and see more global patterns.

(c) **Pooling layers**:
- Reduce spatial dimensions → fewer parameters, faster computation
- Provide some translation invariance
- Increase receptive field of subsequent layers
- Common types: max pooling (keeps strongest activations), average pooling
</details>

---

## Part B: Computation Problems (40 points)

### Question 5 (10 points)
Compute the gradient of:
$$J = z_1 e^{z_1 z_2}, \quad z_1 = a_1 w_1 w_2, \quad z_2 = a_2 w_1 + a_3 w_2^2$$

with respect to $w_1$ and $w_2$.

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Compute partial derivatives of J w.r.t. z₁, z₂
$$\frac{\partial J}{\partial z_1} = e^{z_1 z_2} + z_1 z_2 e^{z_1 z_2} = e^{z_1 z_2}(1 + z_1 z_2)$$
$$\frac{\partial J}{\partial z_2} = z_1^2 e^{z_1 z_2}$$

**Step 2**: Compute partial derivatives of z₁, z₂ w.r.t. w₁, w₂
$$\frac{\partial z_1}{\partial w_1} = a_1 w_2, \quad \frac{\partial z_1}{\partial w_2} = a_1 w_1$$
$$\frac{\partial z_2}{\partial w_1} = a_2, \quad \frac{\partial z_2}{\partial w_2} = 2a_3 w_2$$

**Step 3**: Apply chain rule
$$\frac{\partial J}{\partial w_1} = \frac{\partial J}{\partial z_1}\frac{\partial z_1}{\partial w_1} + \frac{\partial J}{\partial z_2}\frac{\partial z_2}{\partial w_1}$$
$$= e^{z_1 z_2}(1 + z_1 z_2) \cdot a_1 w_2 + z_1^2 e^{z_1 z_2} \cdot a_2$$
$$= e^{z_1 z_2}[a_1 w_2(1 + z_1 z_2) + a_2 z_1^2]$$

$$\frac{\partial J}{\partial w_2} = \frac{\partial J}{\partial z_1}\frac{\partial z_1}{\partial w_2} + \frac{\partial J}{\partial z_2}\frac{\partial z_2}{\partial w_2}$$
$$= e^{z_1 z_2}(1 + z_1 z_2) \cdot a_1 w_1 + z_1^2 e^{z_1 z_2} \cdot 2a_3 w_2$$
$$= e^{z_1 z_2}[a_1 w_1(1 + z_1 z_2) + 2a_3 w_2 z_1^2]$$
</details>

---

### Question 6 (10 points)
Consider a linear SVM classifier with data:

| $x_{i1}$ | 0 | 1 | 1 | 2 |
|----------|---|---|---|---|
| $x_{i2}$ | 0 | 0.5 | 1 | 1 |
| $y_i$ | -1 | -1 | +1 | +1 |

A classifier uses $\hat{y} = \text{sign}(w_1 x_1 + w_2 x_2 + b)$ with $w_1 = 0$, $w_2 = 2$, $b = -1.5$.

(a) Verify that this classifier correctly classifies all points (2 pts)
(b) Compute $z_i = w_1 x_{i1} + w_2 x_{i2} + b$ for each point (2 pts)
(c) Compute the functional margin $\gamma_i = y_i z_i$ for each point (2 pts)
(d) What is the geometric margin $m = \min_i \gamma_i / \|\mathbf{w}\|$? (2 pts)
(e) Which points are support vectors? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Verify classifications**:
- Point 1: z = 0(0) + 2(0) - 1.5 = -1.5 < 0 → ŷ = -1 = y₁ ✓
- Point 2: z = 0(1) + 2(0.5) - 1.5 = -0.5 < 0 → ŷ = -1 = y₂ ✓
- Point 3: z = 0(1) + 2(1) - 1.5 = 0.5 > 0 → ŷ = +1 = y₃ ✓
- Point 4: z = 0(2) + 2(1) - 1.5 = 0.5 > 0 → ŷ = +1 = y₄ ✓

All correct!

(b) **Compute z values** (from above):
- z₁ = -1.5
- z₂ = -0.5
- z₃ = 0.5
- z₄ = 0.5

(c) **Compute functional margins** γᵢ = yᵢzᵢ:
- γ₁ = (-1)(-1.5) = 1.5
- γ₂ = (-1)(-0.5) = 0.5
- γ₃ = (+1)(0.5) = 0.5
- γ₄ = (+1)(0.5) = 0.5

(d) **Geometric margin**:
$$\|\mathbf{w}\| = \sqrt{0^2 + 2^2} = 2$$
$$m = \frac{\min_i \gamma_i}{\|\mathbf{w}\|} = \frac{0.5}{2} = 0.25$$

(e) **Support vectors**: Points 2, 3, and 4 (they achieve the minimum functional margin γ = 0.5)
</details>

---

### Question 7 (10 points)
A neural network has:
- Input: $\mathbf{x} = [x_1, x_2]^T$
- Hidden layer weights and biases:
$$W^H = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}, \quad \mathbf{b}^H = \begin{pmatrix} 0 \\ -1 \end{pmatrix}$$
- Output layer: $W^O = [1, 1]$, $b^O = 0$
- Activation: ReLU for hidden layer, linear for output

For input $\mathbf{x} = [1, 0]^T$:
(a) Compute $\mathbf{z}^H = W^H \mathbf{x} + \mathbf{b}^H$ (2 pts)
(b) Compute $\mathbf{u}^H = \text{ReLU}(\mathbf{z}^H)$ (2 pts)
(c) Compute $z^O$ and $\hat{y}$ (2 pts)
(d) If true label is $y = 2$ and loss is $J = \frac{1}{2}(y - \hat{y})^2$, compute $\frac{\partial J}{\partial \hat{y}}$ (2 pts)
(e) Compute $\frac{\partial J}{\partial \mathbf{u}^H}$ (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Pre-activation hidden**:
$$\mathbf{z}^H = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} + \begin{pmatrix} 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

(b) **Apply ReLU**:
$$\mathbf{u}^H = \text{ReLU}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} \max(0, 1) \\ \max(0, 0) \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

(c) **Output layer**:
$$z^O = W^O \mathbf{u}^H + b^O = [1, 1] \begin{pmatrix} 1 \\ 0 \end{pmatrix} + 0 = 1$$
$$\hat{y} = z^O = 1$$ (linear activation)

(d) **Gradient of loss w.r.t. output**:
$$J = \frac{1}{2}(y - \hat{y})^2 = \frac{1}{2}(2 - 1)^2 = 0.5$$
$$\frac{\partial J}{\partial \hat{y}} = -(y - \hat{y}) = -(2 - 1) = -1$$

(e) **Backprop to hidden activations**:
Since $\hat{y} = z^O$ (linear), $\frac{\partial \hat{y}}{\partial z^O} = 1$

$$\frac{\partial J}{\partial z^O} = \frac{\partial J}{\partial \hat{y}} \cdot 1 = -1$$

$$\frac{\partial J}{\partial \mathbf{u}^H} = (W^O)^T \frac{\partial J}{\partial z^O} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \cdot (-1) = \begin{pmatrix} -1 \\ -1 \end{pmatrix}$$
</details>

---

### Question 8 (10 points)
A convolutional layer has input $X$ of shape $(32, 32, 3)$ (height, width, channels) and kernel $W$ of shape $(5, 5, 3, 16)$.

(a) What is the output shape using valid convolution? (2 pts)
(b) How many parameters does this layer have (including bias)? (2 pts)
(c) How many multiplications are needed to compute the convolution? (3 pts)
(d) If we add a max pooling layer with pool size 2 and stride 2, what is the output shape? (3 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Output shape with valid convolution**:
- Height: 32 - 5 + 1 = 28
- Width: 32 - 5 + 1 = 28
- Channels: 16 (output channels)

**Answer**: $(28, 28, 16)$

(b) **Number of parameters**:
- Weights: $5 \times 5 \times 3 \times 16 = 1200$
- Biases: $16$

**Answer**: $1200 + 16 = 1216$ parameters

(c) **Number of multiplications**:
- For each output pixel: $5 \times 5 \times 3 = 75$ multiplications
- Number of output pixels: $28 \times 28 \times 16 = 12,544$

**Answer**: $75 \times 12,544 = 940,800$ multiplications

(d) **After max pooling (pool size 2, stride 2)**:
- Height: $\lfloor 28/2 \rfloor = 14$
- Width: $\lfloor 28/2 \rfloor = 14$
- Channels: unchanged = 16

**Answer**: $(14, 14, 16)$
</details>

---

## Part C: Proofs and Derivations (30 points)

### Question 9 (10 points)
For gradient descent on $J(\mathbf{w}) = \frac{1}{2}b_1 w_1^2 + \frac{1}{2}b_2 w_2^2$ with $b_2 > b_1 > 0$:

(a) Write the gradient descent update equations (3 pts)
(b) Show that convergence requires $0 < \alpha < \frac{2}{b_2}$ (4 pts)
(c) For step size $\alpha = \frac{2}{b_1 + b_2}$, show the convergence rate is $C = \frac{\kappa - 1}{\kappa + 1}$ where $\kappa = b_2/b_1$ (3 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Gradient and update**:
$$\nabla J = \begin{pmatrix} b_1 w_1 \\ b_2 w_2 \end{pmatrix}$$

Update equations:
$$w_1^{(k+1)} = w_1^{(k)} - \alpha b_1 w_1^{(k)} = (1 - \alpha b_1) w_1^{(k)}$$
$$w_2^{(k+1)} = w_2^{(k)} - \alpha b_2 w_2^{(k)} = (1 - \alpha b_2) w_2^{(k)}$$

(b) **Convergence condition**:
Let $\rho_i = 1 - \alpha b_i$. For convergence, we need $|\rho_i| < 1$ for both i.

For $\rho_1 = 1 - \alpha b_1$:
$$-1 < 1 - \alpha b_1 < 1$$
$$0 < \alpha b_1 < 2$$
$$0 < \alpha < \frac{2}{b_1}$$

For $\rho_2 = 1 - \alpha b_2$:
$$-1 < 1 - \alpha b_2 < 1$$
$$0 < \alpha < \frac{2}{b_2}$$

Since $b_2 > b_1$, we have $\frac{2}{b_2} < \frac{2}{b_1}$, so the binding constraint is:

$$0 < \alpha < \frac{2}{b_2}$$

∎

(c) **Convergence rate**:
With $\alpha = \frac{2}{b_1 + b_2}$:

$$\rho_1 = 1 - \frac{2b_1}{b_1 + b_2} = \frac{b_2 - b_1}{b_1 + b_2}$$

$$\rho_2 = 1 - \frac{2b_2}{b_1 + b_2} = \frac{b_1 - b_2}{b_1 + b_2} = -\frac{b_2 - b_1}{b_1 + b_2}$$

The convergence rate is determined by $\max(|\rho_1|, |\rho_2|)$:

$$C = \frac{b_2 - b_1}{b_1 + b_2} = \frac{b_2/b_1 - 1}{b_2/b_1 + 1} = \frac{\kappa - 1}{\kappa + 1}$$

∎
</details>

---

### Question 10 (10 points)
Derive the backpropagation equations for a single hidden layer network with:
- Hidden layer: $\mathbf{z}^H = W^H \mathbf{x} + \mathbf{b}^H$, $\mathbf{u}^H = \sigma(\mathbf{z}^H)$ (sigmoid)
- Output layer: $z^O = \mathbf{w}^O \cdot \mathbf{u}^H + b^O$, $\hat{y} = z^O$ (linear)
- Loss: $J = \frac{1}{2}(y - \hat{y})^2$

Derive $\frac{\partial J}{\partial W^H}$ and $\frac{\partial J}{\partial \mathbf{b}^H}$.

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Gradient of loss w.r.t. output
$$\frac{\partial J}{\partial \hat{y}} = -(y - \hat{y})$$

**Step 2**: Gradient w.r.t. $z^O$
Since $\hat{y} = z^O$ (linear activation):
$$\frac{\partial J}{\partial z^O} = \frac{\partial J}{\partial \hat{y}} = -(y - \hat{y})$$

**Step 3**: Gradient w.r.t. hidden activations
$$\frac{\partial z^O}{\partial u^H_j} = w^O_j$$

$$\frac{\partial J}{\partial u^H_j} = \frac{\partial J}{\partial z^O} \cdot w^O_j = -(y - \hat{y}) w^O_j$$

In vector form:
$$\frac{\partial J}{\partial \mathbf{u}^H} = -(y - \hat{y}) \mathbf{w}^O$$

**Step 4**: Gradient w.r.t. hidden pre-activations
$$\frac{\partial u^H_j}{\partial z^H_j} = \sigma'(z^H_j) = u^H_j(1 - u^H_j)$$

$$\frac{\partial J}{\partial z^H_j} = \frac{\partial J}{\partial u^H_j} \cdot u^H_j(1 - u^H_j)$$

In vector form (element-wise multiplication):
$$\frac{\partial J}{\partial \mathbf{z}^H} = \frac{\partial J}{\partial \mathbf{u}^H} \odot \mathbf{u}^H \odot (1 - \mathbf{u}^H)$$

**Step 5**: Gradient w.r.t. weights $W^H$
$$z^H_j = \sum_k W^H_{jk} x_k + b^H_j$$
$$\frac{\partial z^H_j}{\partial W^H_{jk}} = x_k$$

$$\frac{\partial J}{\partial W^H_{jk}} = \frac{\partial J}{\partial z^H_j} \cdot x_k$$

In matrix form:
$$\boxed{\frac{\partial J}{\partial W^H} = \frac{\partial J}{\partial \mathbf{z}^H} \mathbf{x}^T}$$

**Step 6**: Gradient w.r.t. bias $\mathbf{b}^H$
$$\frac{\partial z^H_j}{\partial b^H_j} = 1$$

$$\boxed{\frac{\partial J}{\partial \mathbf{b}^H} = \frac{\partial J}{\partial \mathbf{z}^H}}$$
</details>

---

### Question 11 (10 points)
For a 2D convolution layer with sigmoid activation:
$$Z[i,j,m] = \sum_{k_1,k_2,n} W[k_1,k_2,n,m] X[i+k_1,j+k_2,n] + b[m]$$
$$U[i,j,m] = \sigma(Z[i,j,m])$$

Given $\frac{\partial J}{\partial U}$, derive $\frac{\partial J}{\partial W[k_1,k_2,n,m]}$.

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Apply chain rule through sigmoid
$$\frac{\partial J}{\partial Z[i,j,m]} = \frac{\partial J}{\partial U[i,j,m]} \cdot \frac{\partial U[i,j,m]}{\partial Z[i,j,m]}$$

Since $U = \sigma(Z)$:
$$\frac{\partial U}{\partial Z} = \sigma(Z)(1-\sigma(Z)) = U(1-U)$$

Therefore:
$$\frac{\partial J}{\partial Z[i,j,m]} = \frac{\partial J}{\partial U[i,j,m]} \cdot U[i,j,m](1-U[i,j,m])$$

**Step 2**: Gradient w.r.t. weights
From the convolution equation, the weight $W[k_1,k_2,n,m]$ appears in every output $Z[i,j,m]$ (for fixed m):
$$\frac{\partial Z[i,j,m]}{\partial W[k_1,k_2,n,m]} = X[i+k_1,j+k_2,n]$$

Sum over all output positions:
$$\frac{\partial J}{\partial W[k_1,k_2,n,m]} = \sum_i \sum_j \frac{\partial J}{\partial Z[i,j,m]} \cdot \frac{\partial Z[i,j,m]}{\partial W[k_1,k_2,n,m]}$$

$$\boxed{\frac{\partial J}{\partial W[k_1,k_2,n,m]} = \sum_i \sum_j \frac{\partial J}{\partial Z[i,j,m]} \cdot X[i+k_1,j+k_2,n]}$$

This is essentially a convolution between the input X and the gradient $\frac{\partial J}{\partial Z}$.
</details>

---

## Answer Key Summary

| Question | Key Answer |
|----------|------------|
| 6(d) | m = 0.25 |
| 6(e) | Points 2, 3, 4 |
| 7(c) | ŷ = 1 |
| 7(e) | [-1, -1]ᵀ |
| 8(a) | (28, 28, 16) |
| 8(b) | 1216 parameters |
| 8(c) | 940,800 multiplications |
| 8(d) | (14, 14, 16) |

