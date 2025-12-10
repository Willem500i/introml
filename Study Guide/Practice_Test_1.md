# Practice Test 1: Linear Models & Regularization (Units 1-6)

**Time Estimate**: 90 minutes
**Total Points**: 100

---

## Part A: Conceptual Questions (30 points)

### Question 1 (6 points)
Explain the difference between:
(a) Supervised and unsupervised learning (2 pts)
(b) Training error and test error (2 pts)
(c) Bias and variance (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Supervised learning** uses labeled data (input-output pairs) to learn a mapping from inputs to outputs. **Unsupervised learning** finds patterns in unlabeled data without explicit target values.

(b) **Training error** measures how well the model fits the data it was trained on. **Test error** measures how well the model generalizes to new, unseen data. Training error typically underestimates true performance.

(c) **Bias** is error from approximating a complex problem with a simple model (underfitting). **Variance** is error from sensitivity to small fluctuations in the training data (overfitting). There's a tradeoff: increasing model complexity decreases bias but increases variance.
</details>

---

### Question 2 (8 points)
For a linear model $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$, suppose you obtain the following cross-validation results:

| Features Used | Mean CV Train RSS | Mean CV Test RSS | Test RSS Std Dev |
|--------------|-------------------|------------------|------------------|
| $x_1$ only | 5.0 | 5.1 | 0.3 |
| $x_2$ only | 3.5 | 3.6 | 0.4 |
| Both $x_1, x_2$ | 3.2 | 3.4 | 0.5 |

(a) Which model has the best test performance? (2 pts)
(b) Using the "one standard error rule," which model should you select? Show your work. (4 pts)
(c) Why might we prefer a simpler model even if it has slightly worse performance? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) The model with both features has the lowest mean test RSS (3.4), so it has the best test performance.

(b) **One Standard Error Rule**:
- Best model: both features, RSS = 3.4
- Standard error = Std Dev / √k (assume k=10 folds): SE ≈ 0.5/√10 ≈ 0.158
- Threshold: 3.4 + 0.158 = 3.558
- Check simpler models:
  - $x_2$ only: RSS = 3.6 > 3.558 (outside threshold)
  - $x_1$ only: RSS = 5.1 > 3.558 (outside threshold)
- **Select**: Both features model (the simplest model within 1 SE is still the complex model)

(c) Simpler models:
- Are easier to interpret
- Generalize better (lower variance)
- Are more robust to changes in data
- Require less data to train reliably
</details>

---

### Question 3 (8 points)
Consider the LASSO objective:
$$\min_\beta \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|$$

(a) What happens to the coefficients $\beta_j$ as $\lambda \to 0$? As $\lambda \to \infty$? (3 pts)
(b) Why does LASSO produce sparse solutions (some $\beta_j = 0$) while Ridge regression does not? (3 pts)
(c) Why must features be normalized before applying LASSO? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) 
- As $\lambda \to 0$: The penalty vanishes, and LASSO approaches ordinary least squares. All coefficients are non-zero (unless they happen to be zero in OLS).
- As $\lambda \to \infty$: The penalty dominates, forcing all coefficients to zero.

(b) Geometric interpretation: LASSO uses L1 penalty (diamond-shaped constraint region) while Ridge uses L2 penalty (circular region). The diamond has corners at the axes, and the level curves of the RSS often first touch the constraint at these corners (where some coordinates are zero). The circular constraint has no corners, so solutions typically have all non-zero coordinates.

(c) LASSO penalizes $|\beta_j|$ equally for all features. If features have different scales, the penalty affects them differently:
- Large-scale features have small $\beta_j$ → small penalty
- Small-scale features have large $\beta_j$ → large penalty
This creates unfair bias. Normalization puts all features on equal footing.
</details>

---

### Question 4 (8 points)
For logistic regression with $P(y=1|\mathbf{x}) = \frac{1}{1+e^{-z}}$, $z = \boldsymbol{\beta}^T\mathbf{x}$:

(a) What is the gradient of the log-likelihood (or equivalently, the negative gradient of cross-entropy loss)? (3 pts)
(b) Why can't we find a closed-form solution for the optimal $\boldsymbol{\beta}$? (2 pts)
(c) If we multiply all parameters by a constant $\alpha > 1$, how does the decision boundary change? How do the predicted probabilities change? (3 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) The gradient of cross-entropy loss is:
$$\nabla_\beta J = \sum_{i=1}^n (\hat{y}_i - y_i)\mathbf{x}_i = \mathbf{X}^T(\hat{\mathbf{y}} - \mathbf{y})$$
where $\hat{y}_i = \sigma(z_i)$.

(b) Setting $\nabla J = 0$ gives a nonlinear system because $\hat{y}_i = \sigma(\boldsymbol{\beta}^T\mathbf{x}_i)$ is a nonlinear function of $\boldsymbol{\beta}$. Unlike linear regression, we cannot isolate $\boldsymbol{\beta}$ algebraically.

(c) 
- **Decision boundary**: Unchanged! The boundary is where $z = 0$, which means $\boldsymbol{\beta}^T\mathbf{x} = 0$. Multiplying by $\alpha$ gives $(\alpha\boldsymbol{\beta})^T\mathbf{x} = \alpha \cdot 0 = 0$, same boundary.
- **Probabilities**: They become more extreme (closer to 0 or 1). For $z > 0$, $\sigma(\alpha z)$ is larger than $\sigma(z)$ since the sigmoid gets steeper. The model becomes more "confident."
</details>

---

## Part B: Computation Problems (40 points)

### Question 5 (12 points)
Given the following data:

| $x_i$ | 1 | 2 | 3 | 4 | 5 |
|-------|---|---|---|---|---|
| $y_i$ | 2 | 4 | 5 | 4 | 5 |

(a) Compute $\bar{x}$, $\bar{y}$, $s_{xy}$, and $s_x^2$ (4 pts)
(b) Find the least squares estimates $\hat{\beta}_0$ and $\hat{\beta}_1$ for $y = \beta_0 + \beta_1 x$ (4 pts)
(c) What is the predicted value at $x = 6$? (2 pts)
(d) Compute the RSS (residual sum of squares) (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Compute statistics**:
- $\bar{x} = \frac{1+2+3+4+5}{5} = 3$
- $\bar{y} = \frac{2+4+5+4+5}{5} = 4$

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i-\bar{x})(y_i-\bar{y})$ | $(x_i-\bar{x})^2$ |
|-------|-------|-----------------|-----------------|------------------------------|-------------------|
| 1 | 2 | -2 | -2 | 4 | 4 |
| 2 | 4 | -1 | 0 | 0 | 1 |
| 3 | 5 | 0 | 1 | 0 | 0 |
| 4 | 4 | 1 | 0 | 0 | 1 |
| 5 | 5 | 2 | 1 | 2 | 4 |
| Sum | | | | 6 | 10 |

- $s_{xy} = \frac{6}{4} = 1.5$ (using n-1 = 4)
- $s_x^2 = \frac{10}{4} = 2.5$

(b) **Least squares estimates**:
$$\hat{\beta}_1 = \frac{s_{xy}}{s_x^2} = \frac{1.5}{2.5} = 0.6$$
$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x} = 4 - 0.6(3) = 2.2$$

(c) **Prediction at x=6**:
$$\hat{y} = 2.2 + 0.6(6) = 5.8$$

(d) **RSS**:
| $x_i$ | $y_i$ | $\hat{y}_i = 2.2 + 0.6x_i$ | $(y_i - \hat{y}_i)^2$ |
|-------|-------|----------------------------|----------------------|
| 1 | 2 | 2.8 | 0.64 |
| 2 | 4 | 3.4 | 0.36 |
| 3 | 5 | 4.0 | 1.00 |
| 4 | 4 | 4.6 | 0.36 |
| 5 | 5 | 5.2 | 0.04 |
| Sum | | | **2.40** |

RSS = 2.40
</details>

---

### Question 6 (12 points)
Consider fitting a model $\hat{y} = \beta_1\phi_1(x) + \beta_2\phi_2(x)$ with basis functions:
$$\phi_1(x) = \sin(\pi x), \quad \phi_2(x) = \cos(\pi x)$$

Given data points $(x_i, y_i)$: $(0, 1), (0.5, 0), (1, -1)$

(a) Construct the design matrix $\mathbf{A}$ where $A_{ij} = \phi_j(x_i)$ (4 pts)
(b) Compute $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}^T\mathbf{y}$ (4 pts)
(c) Find the least squares solution $\hat{\boldsymbol{\beta}}$ (4 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Design matrix**:
- $\phi_1(0) = \sin(0) = 0$, $\phi_2(0) = \cos(0) = 1$
- $\phi_1(0.5) = \sin(\pi/2) = 1$, $\phi_2(0.5) = \cos(\pi/2) = 0$
- $\phi_1(1) = \sin(\pi) = 0$, $\phi_2(1) = \cos(\pi) = -1$

$$\mathbf{A} = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 0 & -1 \end{pmatrix}$$

(b) **Compute products**:
$$\mathbf{A}^T\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & -1 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 0 & -1 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$$

$$\mathbf{A}^T\mathbf{y} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & -1 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix} = \begin{pmatrix} 0 \\ 2 \end{pmatrix}$$

(c) **Solve for β**:
$$\hat{\boldsymbol{\beta}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{y} = \begin{pmatrix} 1 & 0 \\ 0 & 1/2 \end{pmatrix} \begin{pmatrix} 0 \\ 2 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

So $\hat{y} = \cos(\pi x)$
</details>

---

### Question 7 (8 points)
A logistic regression model has parameters $\beta_0 = -6$, $\beta_1 = 0.05$, $\beta_2 = 1$, where $x_1$ = hours studied and $x_2$ = GPA.

(a) What is the probability that a student with 40 hours studied and GPA 3.5 gets an A? (4 pts)
(b) How many hours would this student need to study to have a 50% chance of getting an A? (4 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Compute z and probability**:
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 = -6 + 0.05(40) + 1(3.5) = -6 + 2 + 3.5 = -0.5$$

$$P(y=1) = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{0.5}} = \frac{1}{1+1.649} \approx 0.378$$

**Answer**: About 37.8% probability

(b) **Find hours for 50% probability**:
$P = 0.5$ means $z = 0$:
$$0 = -6 + 0.05 x_1 + 1(3.5)$$
$$0 = -6 + 0.05 x_1 + 3.5$$
$$0.05 x_1 = 2.5$$
$$x_1 = 50$$

**Answer**: 50 hours of studying
</details>

---

### Question 8 (8 points)
Given normalized features with LASSO estimates $\alpha_1 = 0.6$, $\alpha_2 = -0.3$ for model:
$$\hat{u} = \alpha_1 z_1 + \alpha_2 z_2$$
where $z_j = \frac{x_j - \bar{x}_j}{s_j}$ and $u = \frac{y - \bar{y}}{s_y}$

The original data has:
- $\bar{x}_1 = 50000$, $s_1 = 15000$ (income in dollars)
- $\bar{x}_2 = 45$, $s_2 = 10$ (age in years)
- $\bar{y} = 300$, $s_y = 100$ (house price in $1000s)

Convert to original coordinates: $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Convert coefficients
$$\beta_1 = \frac{s_y \alpha_1}{s_1} = \frac{100 \times 0.6}{15000} = 0.004$$
$$\beta_2 = \frac{s_y \alpha_2}{s_2} = \frac{100 \times (-0.3)}{10} = -3$$

**Step 2**: Compute intercept
$$\beta_0 = \bar{y} - \beta_1 \bar{x}_1 - \beta_2 \bar{x}_2$$
$$= 300 - 0.004(50000) - (-3)(45)$$
$$= 300 - 200 + 135 = 235$$

**Answer**: $\hat{y} = 235 + 0.004 x_1 - 3 x_2$

Or: House price ($1000s) = 235 + 0.004 × Income($) - 3 × Age(years)
</details>

---

## Part C: Proofs (30 points)

### Question 9 (10 points)
Prove that for simple linear regression $y = \beta_0 + \beta_1 x + \epsilon$, the least squares estimate is:
$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Write the RSS
$$RSS = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$$

**Step 2**: Take partial derivatives
$$\frac{\partial RSS}{\partial \beta_0} = -2\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0$$
$$\frac{\partial RSS}{\partial \beta_1} = -2\sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1 x_i) = 0$$

**Step 3**: From first equation
$$\sum_{i=1}^n y_i = n\beta_0 + \beta_1 \sum_{i=1}^n x_i$$
$$\bar{y} = \beta_0 + \beta_1 \bar{x}$$
$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

**Step 4**: Substitute into second equation
$$\sum_{i=1}^n x_i y_i = \beta_0 \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2$$
$$\sum_{i=1}^n x_i y_i = (\bar{y} - \beta_1 \bar{x})(n\bar{x}) + \beta_1 \sum_{i=1}^n x_i^2$$
$$\sum_{i=1}^n x_i y_i = n\bar{x}\bar{y} - \beta_1 n\bar{x}^2 + \beta_1 \sum_{i=1}^n x_i^2$$
$$\sum_{i=1}^n x_i y_i - n\bar{x}\bar{y} = \beta_1 \left(\sum_{i=1}^n x_i^2 - n\bar{x}^2\right)$$

**Step 5**: Recognize the formula
- LHS: $\sum_i x_i y_i - n\bar{x}\bar{y} = \sum_i (x_i - \bar{x})(y_i - \bar{y})$
- RHS: $\sum_i x_i^2 - n\bar{x}^2 = \sum_i (x_i - \bar{x})^2$

Therefore:
$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$ ∎
</details>

---

### Question 10 (10 points)
For a linear model with no intercept, $y = \beta x$, derive the least squares estimate:
$$\hat{\beta} = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2}$$

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Write the RSS
$$RSS(\beta) = \sum_{i=1}^n (y_i - \beta x_i)^2$$

**Step 2**: Expand
$$RSS = \sum_{i=1}^n (y_i^2 - 2\beta x_i y_i + \beta^2 x_i^2)$$
$$= \sum_{i=1}^n y_i^2 - 2\beta \sum_{i=1}^n x_i y_i + \beta^2 \sum_{i=1}^n x_i^2$$

**Step 3**: Take derivative
$$\frac{dRSS}{d\beta} = -2\sum_{i=1}^n x_i y_i + 2\beta \sum_{i=1}^n x_i^2$$

**Step 4**: Set to zero and solve
$$-2\sum_{i=1}^n x_i y_i + 2\beta \sum_{i=1}^n x_i^2 = 0$$
$$\beta \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i$$
$$\hat{\beta} = \frac{\sum_{i=1}^n x_i y_i}{\sum_{i=1}^n x_i^2}$$ ∎

**Step 5**: Verify it's a minimum
$$\frac{d^2 RSS}{d\beta^2} = 2\sum_{i=1}^n x_i^2 > 0$$

The second derivative is positive (assuming not all $x_i = 0$), confirming this is a minimum.
</details>

---

### Question 11 (10 points)
Show that for the soft-thresholding problem:
$$\hat{w} = \arg\min_w \frac{1}{2}(y-w)^2 + \lambda|w|$$

the solution is:
$$\hat{w} = \text{sign}(y) \max(|y| - \lambda, 0)$$

<details>
<summary><strong>Solution</strong></summary>

**Case 1: Assume $w > 0$**

Then $|w| = w$, and:
$$J(w) = \frac{1}{2}(y-w)^2 + \lambda w$$
$$J'(w) = -(y-w) + \lambda = w - y + \lambda$$

Setting $J'(w) = 0$: $w = y - \lambda$

This is valid only if $w > 0$, i.e., $y > \lambda$.

**Case 2: Assume $w < 0$**

Then $|w| = -w$, and:
$$J(w) = \frac{1}{2}(y-w)^2 - \lambda w$$
$$J'(w) = -(y-w) - \lambda = w - y - \lambda$$

Setting $J'(w) = 0$: $w = y + \lambda$

This is valid only if $w < 0$, i.e., $y < -\lambda$.

**Case 3: Neither case applies (i.e., $|y| \leq \lambda$)**

Check $w = 0$: For $J$ to have minimum at $w = 0$, we need:
- Derivative from right: $J'(0^+) = -y + \lambda \geq 0 \Rightarrow y \leq \lambda$
- Derivative from left: $J'(0^-) = -y - \lambda \leq 0 \Rightarrow y \geq -\lambda$

Both conditions satisfied when $|y| \leq \lambda$.

**Combining all cases**:
$$\hat{w} = \begin{cases} y - \lambda & y > \lambda \\ 0 & |y| \leq \lambda \\ y + \lambda & y < -\lambda \end{cases}$$

This can be written compactly as:
$$\hat{w} = \text{sign}(y) \max(|y| - \lambda, 0)$$ ∎
</details>

---

## Answer Key Summary

| Question | Key Answer |
|----------|------------|
| 5(b) | $\hat{\beta}_0 = 2.2$, $\hat{\beta}_1 = 0.6$ |
| 5(c) | $\hat{y}(6) = 5.8$ |
| 5(d) | RSS = 2.40 |
| 6(c) | $\hat{\beta}_1 = 0$, $\hat{\beta}_2 = 1$ |
| 7(a) | P ≈ 0.378 (37.8%) |
| 7(b) | 50 hours |
| 8 | $\hat{y} = 235 + 0.004x_1 - 3x_2$ |

