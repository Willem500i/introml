# ML Quick Reference Card

## Essential Formulas

### Linear Regression
```
Least Squares:  β̂ = (X'X)⁻¹X'y
Simple:         β̂₁ = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
                β̂₀ = ȳ - β̂₁x̄
No intercept:   β̂ = Σxᵢyᵢ / Σxᵢ²
```

### Regularization
```
Ridge:   β̂ = (X'X + λI)⁻¹X'y
LASSO:   min ||y - Xβ||² + λ||β||₁  (use coordinate descent)

Soft thresholding:
  ŵ = sign(y)·max(|y|-λ, 0)
```

### Logistic Regression
```
P(y=1|x) = σ(z) = 1/(1+e⁻ᶻ), z = β'x

Loss:     J = Σ[log(1+eᶻⁱ) - yᵢzᵢ]
Gradient: ∇J = X'(ŷ - y) where ŷᵢ = σ(zᵢ)
```

### Gradient Descent
```
Update:  w⁽ᵏ⁺¹⁾ = w⁽ᵏ⁾ - α∇J(w⁽ᵏ⁾)

Convergence for quadratic J = ½b₁w₁² + ½b₂w₂²:
  Requires: 0 < α < 2/b_max
  Rate: C = (κ-1)/(κ+1), κ = b_max/b_min
```

### SVM
```
Margin:     m = γ/||w||
Hinge loss: εᵢ = max(0, 1 - yᵢzᵢ)

Decision:   ŷ = sign(w'x + b)
Kernel:     z = Σαᵢyᵢ K(xᵢ, x)

Common kernels:
  RBF: K(x,x') = exp(-γ||x-x'||²)
  Poly: K(x,x') = (1 + x'x')^d
```

### Neural Networks
```
Forward:  z = Wx + b
          u = g(z)  (activation)

Activations:
  ReLU:    g(z) = max(0,z),    g'(z) = 𝟙{z>0}
  Sigmoid: g(z) = 1/(1+e⁻ᶻ),  g'(z) = g(g)(1-g(z))
  
Backprop (sigmoid):
  ∂J/∂z = ∂J/∂u ⊙ u ⊙ (1-u)
  ∂J/∂W = ∂J/∂z · x'
  ∂J/∂x = W' · ∂J/∂z  (to previous layer)
```

### CNN
```
Output size (valid): (H - K + 1) × (W - K + 1)
Parameters: K_H × K_W × C_in × C_out + C_out (bias)
Multiplications: output_size × K_H × K_W × C_in

Pooling (stride s, pool p):
  Output size: floor(input_size / s)
```

### PCA
```
Q = X̃'X̃/(n-1)        (covariance of centered data)
Q = VΛV'              (eigendecomposition)

PC coefficients: z = V'(x - μ)
Reconstruction:  x̂ = μ + V_k z_k  (using k PCs)
Error: ||x - x̂||² = Σⱼ₌ₖ₊₁ zⱼ²

Variance explained: PVE_k = Σⱼ₌₁ᵏλⱼ / Σλⱼ
```

### Clustering
```
K-means objective: J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²

Update:  μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
Assign:  cᵢ = argminₖ ||xᵢ - μₖ||²
```

### Decision Trees
```
Gini:   G = Σₖ pₖ(1-pₖ) = 1 - Σₖ pₖ²
Entropy: H = -Σₖ pₖ log(pₖ)

Info Gain = H(parent) - Σ (nchild/nparent)·H(child)
```

---

## Key Concepts Checklist

### Supervised Learning
- [ ] Regression vs Classification
- [ ] Training vs Test error
- [ ] Bias-Variance tradeoff
- [ ] Cross-validation

### Linear Models
- [ ] Normal equations derivation
- [ ] Feature transforms (basis functions)
- [ ] One-hot encoding

### Regularization
- [ ] Ridge vs LASSO (L2 vs L1)
- [ ] Why LASSO is sparse
- [ ] Normalization before regularization

### Model Selection
- [ ] K-fold CV
- [ ] One standard error rule
- [ ] Underfitting vs overfitting

### Logistic Regression
- [ ] Sigmoid function properties
- [ ] Cross-entropy loss
- [ ] Decision boundary

### Optimization
- [ ] Gradient descent
- [ ] Chain rule for gradients
- [ ] Condition number
- [ ] Local vs global minima

### SVMs
- [ ] Margin maximization
- [ ] Support vectors
- [ ] Hinge loss
- [ ] Kernel trick

### Neural Networks
- [ ] Forward propagation
- [ ] Backpropagation
- [ ] Activation functions
- [ ] Computing number of parameters

### CNNs
- [ ] Convolution operation
- [ ] Tensor shapes
- [ ] Pooling
- [ ] Parameter sharing

### PCA
- [ ] Eigenvalue = variance
- [ ] Reconstruction error
- [ ] Choosing number of PCs

### Clustering
- [ ] K-means algorithm
- [ ] Convergence proof
- [ ] Choosing K

### Decision Trees
- [ ] Splitting criteria
- [ ] Overfitting and pruning
- [ ] Random forests

---

## Common Proof Strategies

1. **Least Squares Derivation**
   - Write RSS, expand, differentiate, set to zero

2. **Gradient Calculations**
   - Identify intermediate variables
   - Apply chain rule step by step
   - Be careful with matrix dimensions

3. **Showing Bias = 0**
   - Substitute true model into estimator
   - Take expectation
   - Show E[β̂] = β₀

4. **Convergence Proofs**
   - Show objective decreases at each step
   - Show objective is bounded below
   - Conclude convergence

---

## Exam Tips

1. **Read carefully** - identify what type of problem (proof, calculation, concept)

2. **Show your work** - partial credit often available

3. **Check dimensions** - matrix multiplication must be compatible

4. **Units matter** - especially for normalization problems

5. **Verify answers** - plug back in to check if reasonable

6. **Manage time** - don't get stuck on one problem

7. **Start with what you know** - build confidence then tackle harder parts

