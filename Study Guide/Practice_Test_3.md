# Practice Test 3: PCA, Clustering, and Decision Trees (Units 11-13)

**Time Estimate**: 60 minutes
**Total Points**: 75

---

## Part A: Conceptual Questions (25 points)

### Question 1 (8 points)
For Principal Component Analysis:
(a) What is the relationship between eigenvalues and variance explained? (2 pts)
(b) How do you determine how many PCs to keep? (2 pts)
(c) What is the connection between PCA and SVD? (2 pts)
(d) Why must you center the data before computing PCA? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Eigenvalues and variance**: The eigenvalue λⱼ of the covariance matrix equals the variance of the data projected onto the j-th principal component. Larger eigenvalue = more variance captured by that PC.

(b) **Choosing number of PCs**:
- Proportion of variance explained (PVE): Keep enough PCs for PVE > threshold (e.g., 90%)
- Scree plot: Look for "elbow" where eigenvalues drop sharply
- Cross-validation: If using PCs for prediction, validate on held-out data

(c) **PCA-SVD connection**: For centered data matrix X̃ = UΣVᵀ:
- Principal components: columns of V
- PC coefficients: Z = UΣ
- Eigenvalues of covariance: λⱼ = σⱼ²/(n-1)

(d) **Why center**: PCA finds directions of maximum variance. Without centering, the first PC would point toward the mean of the data rather than capturing the structure. The covariance matrix is defined for centered data: Q = X̃ᵀX̃/(n-1).
</details>

---

### Question 2 (9 points)
For K-means clustering:
(a) Write the K-means objective function. What does it measure? (3 pts)
(b) Is K-means guaranteed to converge? Does it find the global optimum? (3 pts)
(c) How does the choice of K affect the results? How can you choose K? (3 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **K-means objective**:
$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

It measures the total within-cluster sum of squares—the sum of squared distances from each point to its assigned cluster center. Lower J means tighter, more compact clusters.

(b) **Convergence**:
- **Yes, K-means always converges** because each step (assignment and update) either decreases J or keeps it the same, and J is bounded below by 0.
- **No, not guaranteed global optimum**: K-means can get stuck in local minima depending on initialization. Common solutions: run multiple times with random initialization, or use K-means++.

(c) **Effect of K and choosing it**:
- Small K: Underfits, large heterogeneous clusters
- Large K: Overfits, small clusters may be meaningless
- **Elbow method**: Plot J vs K, look for diminishing returns
- **Silhouette score**: Measures cluster cohesion vs separation
- **Gap statistic**: Compare J to null distribution
- Domain knowledge about expected number of groups
</details>

---

### Question 3 (8 points)
For Decision Trees:
(a) What splitting criteria can be used for classification? How do they differ? (3 pts)
(b) What is overfitting in decision trees and how can it be prevented? (3 pts)
(c) How does a Random Forest improve upon a single decision tree? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Splitting criteria for classification**:
- **Misclassification rate**: 1 - max_k(pₖ). Simple but not smooth—doesn't differentiate between nodes with same majority class
- **Gini impurity**: Σpₖ(1-pₖ). Measures probability of wrong classification with random guess. Prefers splits creating pure nodes
- **Entropy/Information gain**: -Σpₖlog(pₖ). Information-theoretic measure. Similar to Gini in practice

(b) **Overfitting and prevention**:
- **Overfitting**: Deep trees can fit noise, creating many leaves that don't generalize. Each leaf may have very few samples.
- **Prevention**:
  - Pre-pruning: Set max depth, min samples per leaf, min samples for split
  - Post-pruning: Grow full tree, then remove nodes that don't improve validation error
  - Use cross-validation to select hyperparameters

(c) **Random Forest advantages**:
- Reduces variance through averaging multiple trees (bagging)
- Each tree trained on bootstrap sample (different subset)
- Random feature selection at each split increases diversity
- More robust to overfitting than single deep tree
- Can handle high-dimensional data well
</details>

---

## Part B: Computation Problems (30 points)

### Question 4 (15 points)
Given data matrix:
$$X = \begin{pmatrix} 3 & 2 \\ 2 & 4 \\ 1 & 2 \\ 0 & 2 \end{pmatrix}$$

(a) Compute the sample mean $\boldsymbol{\mu}$ (2 pts)
(b) Compute the centered data matrix $\tilde{X}$ (2 pts)
(c) Compute the covariance matrix $Q$ (4 pts)
(d) Find the eigenvalues and eigenvectors of Q (3 pts)
(e) What proportion of variance is explained by the first PC? (2 pts)
(f) Project the first sample onto the first PC (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Sample mean**:
$$\boldsymbol{\mu} = \frac{1}{4}\begin{pmatrix} 3+2+1+0 \\ 2+4+2+2 \end{pmatrix} = \begin{pmatrix} 1.5 \\ 2.5 \end{pmatrix}$$

(b) **Centered data**:
$$\tilde{X} = X - \mathbf{1}\boldsymbol{\mu}^T = \begin{pmatrix} 3-1.5 & 2-2.5 \\ 2-1.5 & 4-2.5 \\ 1-1.5 & 2-2.5 \\ 0-1.5 & 2-2.5 \end{pmatrix} = \begin{pmatrix} 1.5 & -0.5 \\ 0.5 & 1.5 \\ -0.5 & -0.5 \\ -1.5 & -0.5 \end{pmatrix}$$

(c) **Covariance matrix**:
$$Q = \frac{1}{n-1}\tilde{X}^T\tilde{X} = \frac{1}{3}\tilde{X}^T\tilde{X}$$

$$\tilde{X}^T\tilde{X} = \begin{pmatrix} 1.5 & 0.5 & -0.5 & -1.5 \\ -0.5 & 1.5 & -0.5 & -0.5 \end{pmatrix} \begin{pmatrix} 1.5 & -0.5 \\ 0.5 & 1.5 \\ -0.5 & -0.5 \\ -1.5 & -0.5 \end{pmatrix}$$

$$= \begin{pmatrix} 1.5^2+0.5^2+0.5^2+1.5^2 & -0.75+0.75+0.25+0.75 \\ -0.75+0.75+0.25+0.75 & 0.25+2.25+0.25+0.25 \end{pmatrix}$$

$$= \begin{pmatrix} 5 & 1 \\ 1 & 3 \end{pmatrix}$$

$$Q = \frac{1}{3}\begin{pmatrix} 5 & 1 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 5/3 & 1/3 \\ 1/3 & 1 \end{pmatrix}$$

(d) **Eigenvalues**:
$$\det(Q - \lambda I) = (5/3 - \lambda)(1 - \lambda) - 1/9 = 0$$
$$5/3 - 5\lambda/3 - \lambda + \lambda^2 - 1/9 = 0$$
$$\lambda^2 - \frac{8}{3}\lambda + \frac{14}{9} = 0$$

Using quadratic formula:
$$\lambda = \frac{8/3 \pm \sqrt{64/9 - 56/9}}{2} = \frac{8/3 \pm \sqrt{8/9}}{2} = \frac{8/3 \pm 2\sqrt{2}/3}{2}$$

$$\lambda_1 = \frac{4 + \sqrt{2}}{3} \approx 1.805, \quad \lambda_2 = \frac{4 - \sqrt{2}}{3} \approx 0.862$$

**Eigenvectors** (can verify by substitution):
- $\mathbf{v}_1 \propto [1, 0.382]^T$ (normalized)
- $\mathbf{v}_2 \propto [-0.382, 1]^T$ (normalized)

(e) **Proportion of variance by first PC**:
$$PVE_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{1.805}{1.805 + 0.862} = \frac{1.805}{2.667} \approx 0.677$$

About 67.7% of variance explained.

(f) **Project first sample**:
First centered sample: $\tilde{\mathbf{x}}_1 = [1.5, -0.5]^T$

$$z_1 = \mathbf{v}_1^T \tilde{\mathbf{x}}_1$$

With normalized $\mathbf{v}_1 \approx [0.934, 0.357]^T$:
$$z_1 \approx 0.934(1.5) + 0.357(-0.5) \approx 1.401 - 0.179 \approx 1.22$$
</details>

---

### Question 5 (8 points)
Given 5 data points: $(0,0), (1,0), (0,1), (3,3), (4,3)$

Starting with K=2 initial centers at $(0,0)$ and $(3,3)$:
(a) Assign each point to the nearest center (3 pts)
(b) Compute the new cluster centers (3 pts)
(c) Has K-means converged? If not, what happens next? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Distance calculations and assignments**:

| Point | Dist to (0,0) | Dist to (3,3) | Assign |
|-------|---------------|---------------|--------|
| (0,0) | 0 | √18 ≈ 4.24 | C1 |
| (1,0) | 1 | √13 ≈ 3.61 | C1 |
| (0,1) | 1 | √13 ≈ 3.61 | C1 |
| (3,3) | √18 ≈ 4.24 | 0 | C2 |
| (4,3) | 5 | 1 | C2 |

**Clusters**:
- C1: {(0,0), (1,0), (0,1)}
- C2: {(3,3), (4,3)}

(b) **New centers**:
$$\boldsymbol{\mu}_1^{new} = \frac{1}{3}[(0,0) + (1,0) + (0,1)] = \left(\frac{1}{3}, \frac{1}{3}\right) \approx (0.33, 0.33)$$

$$\boldsymbol{\mu}_2^{new} = \frac{1}{2}[(3,3) + (4,3)] = (3.5, 3)$$

(c) **Convergence check**:
Centers changed from (0,0)→(0.33,0.33) and (3,3)→(3.5,3), so **not converged**.

Next iteration:
- Reassign points using new centers
- Likely same assignments since clusters are well-separated
- If assignments don't change, algorithm converges
</details>

---

### Question 6 (7 points)
Consider a decision tree for binary classification with the following data:

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

(a) Compute the Gini impurity of the root node (2 pts)
(b) Compute the Gini impurity after splitting on $x_1 = 0.5$ (3 pts)
(c) What is the information gain of this split? (2 pts)

<details>
<summary><strong>Solution</strong></summary>

(a) **Root Gini impurity**:
At root: 2 samples with y=0, 2 samples with y=1
$$p_0 = 2/4 = 0.5, \quad p_1 = 2/4 = 0.5$$
$$Gini_{root} = 2 \cdot p_0 \cdot p_1 = 2 \cdot 0.5 \cdot 0.5 = 0.5$$

(Or equivalently: $1 - p_0^2 - p_1^2 = 1 - 0.25 - 0.25 = 0.5$)

(b) **After split on $x_1 = 0.5$**:

Left child ($x_1 < 0.5$): points (0,0) and (0,1), both y=0
- $p_0 = 1, p_1 = 0$
- $Gini_{left} = 2 \cdot 1 \cdot 0 = 0$ (pure node)

Right child ($x_1 \geq 0.5$): points (1,0) and (1,1), both y=1
- $p_0 = 0, p_1 = 1$
- $Gini_{right} = 2 \cdot 0 \cdot 1 = 0$ (pure node)

**Weighted average**:
$$Gini_{after} = \frac{2}{4}(0) + \frac{2}{4}(0) = 0$$

(c) **Information gain** (Gini-based):
$$IG = Gini_{root} - Gini_{after} = 0.5 - 0 = 0.5$$

This is the maximum possible gain—the split perfectly separates the classes!
</details>

---

## Part C: Proofs (20 points)

### Question 7 (10 points)
Prove that the reconstruction error in PCA using k principal components equals the sum of the squared PC coefficients for the skipped components.

That is, show: $\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{j=k+1}^d z_j^2$

where $\hat{\mathbf{x}} = \boldsymbol{\mu} + \sum_{j=1}^k z_j \mathbf{v}_j$ and $z_j = \mathbf{v}_j^T(\mathbf{x} - \boldsymbol{\mu})$.

<details>
<summary><strong>Solution</strong></summary>

**Step 1**: Express centered data in PC basis
Since the PC vectors $\{\mathbf{v}_1, ..., \mathbf{v}_d\}$ form an orthonormal basis, we can write:
$$\mathbf{x} - \boldsymbol{\mu} = \sum_{j=1}^d z_j \mathbf{v}_j$$

where $z_j = \mathbf{v}_j^T(\mathbf{x} - \boldsymbol{\mu})$ are the PC coefficients.

**Step 2**: Write reconstruction using k components
$$\hat{\mathbf{x}} - \boldsymbol{\mu} = \sum_{j=1}^k z_j \mathbf{v}_j$$

**Step 3**: Compute reconstruction error
$$\mathbf{x} - \hat{\mathbf{x}} = (\mathbf{x} - \boldsymbol{\mu}) - (\hat{\mathbf{x}} - \boldsymbol{\mu})$$
$$= \sum_{j=1}^d z_j \mathbf{v}_j - \sum_{j=1}^k z_j \mathbf{v}_j$$
$$= \sum_{j=k+1}^d z_j \mathbf{v}_j$$

**Step 4**: Compute squared norm
$$\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \left\|\sum_{j=k+1}^d z_j \mathbf{v}_j\right\|^2$$

**Step 5**: Use orthonormality
Since $\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}$ (Kronecker delta):
$$\left\|\sum_{j=k+1}^d z_j \mathbf{v}_j\right\|^2 = \left(\sum_{j=k+1}^d z_j \mathbf{v}_j\right)^T \left(\sum_{i=k+1}^d z_i \mathbf{v}_i\right)$$
$$= \sum_{j=k+1}^d \sum_{i=k+1}^d z_j z_i (\mathbf{v}_j^T \mathbf{v}_i)$$
$$= \sum_{j=k+1}^d z_j^2$$

Therefore:
$$\boxed{\|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \sum_{j=k+1}^d z_j^2}$$ ∎
</details>

---

### Question 8 (10 points)
Show that the K-means algorithm monotonically decreases the objective function:
$$J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

at each step (assignment and update), and therefore must converge.

<details>
<summary><strong>Solution</strong></summary>

**Part 1: Assignment step decreases (or maintains) J**

In the assignment step, each point $\mathbf{x}_i$ is assigned to its nearest center. Let $c_i^{old}$ be the old assignment and $c_i^{new}$ be the new assignment.

By definition of nearest:
$$\|\mathbf{x}_i - \boldsymbol{\mu}_{c_i^{new}}\|^2 \leq \|\mathbf{x}_i - \boldsymbol{\mu}_{c_i^{old}}\|^2$$

Since this holds for every point, the total objective:
$$J^{new} = \sum_i \|\mathbf{x}_i - \boldsymbol{\mu}_{c_i^{new}}\|^2 \leq \sum_i \|\mathbf{x}_i - \boldsymbol{\mu}_{c_i^{old}}\|^2 = J^{old}$$

**Part 2: Update step decreases (or maintains) J**

In the update step, for fixed assignments, we update each center:
$$\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$$

For fixed cluster $C_k$, consider minimizing:
$$J_k = \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}\|^2$$

over $\boldsymbol{\mu}$. Taking the gradient:
$$\nabla_\mu J_k = -2\sum_{i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}) = 0$$
$$\sum_{i \in C_k} \mathbf{x}_i = |C_k| \boldsymbol{\mu}$$
$$\boldsymbol{\mu} = \frac{1}{|C_k|}\sum_{i \in C_k} \mathbf{x}_i$$

This is exactly the update formula! So the new center minimizes $J_k$, meaning:
$$J_k^{new} \leq J_k^{old}$$

Since this holds for all clusters:
$$J^{new} = \sum_k J_k^{new} \leq \sum_k J_k^{old} = J^{old}$$

**Part 3: Convergence**

- J is bounded below by 0 (sum of non-negative terms)
- J decreases (or stays same) at each step
- J can only take finitely many values (finite # of possible assignments)

Therefore, J must eventually stop decreasing, which means the algorithm has converged. ∎

Note: Convergence is to a local minimum, not necessarily the global minimum.
</details>

---

## Answer Key Summary

| Question | Key Answer |
|----------|------------|
| 4(a) | μ = [1.5, 2.5]ᵀ |
| 4(c) | Q = [[5/3, 1/3], [1/3, 1]] |
| 4(e) | ~67.7% |
| 5(b) | μ₁ = (0.33, 0.33), μ₂ = (3.5, 3) |
| 6(a) | Gini = 0.5 |
| 6(b) | Gini = 0 |
| 6(c) | IG = 0.5 |

