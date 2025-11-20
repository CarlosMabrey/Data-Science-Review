import { ExplanationContent } from '../types';

// This file acts as a robust "Goldmine" of expert-verified content.
// It ensures instant load times and high-quality formatting for every topic in the app.

export const STATIC_CONTENT: Record<string, ExplanationContent> = {
  // --- EXISTING ---
  'lin-reg': {
    overview: `**Linear Regression** models the relationship between a scalar response and one or more explanatory variables using a linear approach. It assumes the target $y$ is a linear combination of inputs $\\mathbf{x}$ plus irreducible error.`,
    mathematicalIntuition: `
The model is defined as:
$$ y = \\beta_0 + \\beta_1 x_1 + \\dots + \\beta_p x_p + \\epsilon $$
We minimize the **Residual Sum of Squares (RSS)**:
$$ J(\\boldsymbol{\\beta}) = \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2 $$
The closed-form solution (Normal Equation) is:
$$ \\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{y} $$
    `,
    useCases: ["Algorithmic Trading (Beta estimation)", "Real Estate Valuation", "Marketing Mix Modeling"],
    prosCons: {
        pros: ["Highly Interpretable", "Analytic Solution exists", "Basis for GLMs"],
        cons: ["Assumes Linearity", "Sensitive to Outliers", "Multicollinearity issues"]
    },
    codeSnippet: `import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Plot
X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.", label="Data")
plt.legend()
plt.show()`,
    quiz: [
       { question: "What does the slope coefficient represent?", options: ["Correlation", "Change in Y per unit X", "Variance", "Bias"], correctIndex: 1, explanation: "It represents the marginal effect of X on Y." },
       { question: "Which assumption is NOT required for OLS?", options: ["Linearity", "Homoscedasticity", "Normality of Predictors", "Independence of Errors"], correctIndex: 2, explanation: "The *residuals* should be normal for inference, but the predictors (X) do not need to be normal." }
    ],
    deepDive: {
        advancedTheory: "Gauss-Markov Theorem states OLS is BLUE (Best Linear Unbiased Estimator).",
        keyFormulas: ["\\hat{\\beta} = (X^TX)^{-1}X^Ty"],
        seminalPapers: []
    }
  },
  'k-means': {
    overview: `**K-Means** partitions $n$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean (centroid). It is a prototype-based clustering method.`,
    mathematicalIntuition: `
Objective: Minimize Within-Cluster Sum of Squares (WCSS):
$$ J = \\sum_{i=1}^{k} \\sum_{\\mathbf{x} \\in S_i} \\| \\mathbf{x} - \\boldsymbol{\\mu}_i \\|^2 $$
Algorithm (Lloyd's):
1. Assign points to nearest centroid.
2. Update centroid to mean of assigned points.
3. Repeat until convergence.
    `,
    useCases: ["Customer Segmentation", "Image Compression", "Anomaly Detection"],
    prosCons: {
        pros: ["Scalable $O(N)$", "Simple implementation", "Converges to local optima"],
        cons: ["Must specify K", "Sensitive to initialization", "Assumes spherical clusters"]
    },
    codeSnippet: `from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 2)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200)
plt.show()`,
    quiz: [
        { question: "What is the Elbow Method used for?", options: ["Initializing centroids", "Finding optimal K", "Regularization", "Gradient Descent"], correctIndex: 1, explanation: "It identifies the K where diminishing returns in WCSS reduction set in." }
    ],
    deepDive: {
        advancedTheory: "K-Means is a hard-assignment limit of Gaussian Mixture Models (GMM).",
        keyFormulas: ["\\mu_i^{(t+1)} = \\frac{1}{|S_i^{(t)}|} \\sum_{x_j \\in S_i^{(t)}} x_j"],
        seminalPapers: []
    }
  },
  'logistic-reg': {
    overview: `**Logistic Regression** is a probabilistic classifier that models the probability of a binary outcome using the sigmoid function. Despite its name, it is used for classification, not regression.`,
    mathematicalIntuition: `
We apply the sigmoid activation $\\sigma(z) = \\frac{1}{1+e^{-z}}$ to the linear predictor $z = \\mathbf{w}^T\\mathbf{x} + b$.
$$ P(y=1|x) = \\sigma(\\mathbf{w}^T\\mathbf{x} + b) $$
We minimize the **Log Loss** (Cross-Entropy):
$$ J(\\mathbf{w}) = -\\frac{1}{m} \\sum [y^{(i)} \\log(\\hat{y}^{(i)}) + (1-y^{(i)}) \\log(1-\\hat{y}^{(i)})] $$
    `,
    useCases: ["Spam Detection", "Credit Default Prediction", "Medical Diagnosis"],
    prosCons: {
        pros: ["Probabilistic output", "Easy to regularize", "Feature importance via weights"],
        cons: ["Linear decision boundary", "Prone to overfitting if N < P", "Cannot handle non-linearities directly"]
    },
    codeSnippet: `from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
clf = LogisticRegression().fit(X, y)

# Decision Boundary
xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
plt.scatter(X[:,0], X[:, 1], c=y, edgecolor="white")
plt.show()`,
    quiz: [
        { question: "What is the range of the sigmoid function?", options: ["[-1, 1]", "[0, 1]", "[0, infinity]", "[-infinity, infinity]"], correctIndex: 1, explanation: "Sigmoid maps any real number to the probability interval [0, 1]." }
    ],
    deepDive: {
        advancedTheory: "Logistic Regression is a Generalized Linear Model (GLM) with a Bernoulli distribution and Logit link function.",
        keyFormulas: ["\\sigma(z) = \\frac{1}{1+e^{-z}}", "\\frac{\\partial J}{\\partial w_j} = (\\hat{y} - y)x_j"],
        seminalPapers: []
    }
  },
  'gradient-descent': {
    overview: `**Gradient Descent** is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. It is the workhorse of modern machine learning, especially Deep Learning.`,
    mathematicalIntuition: `
To minimize a cost function $J(\\theta)$, we update parameters in the opposite direction of the gradient:
$$ \\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta J(\\theta_t) $$
Where $\\eta$ is the **learning rate**.
Types:
*   **Batch GD:** Uses all data for gradient.
*   **Stochastic GD (SGD):** Uses one sample.
*   **Mini-batch GD:** Uses a subset ($32-256$ samples).
    `,
    useCases: ["Training Neural Networks", "Optimizing Linear/Logistic Regression", "Matrix Factorization"],
    prosCons: {
        pros: ["General purpose", "Scales to large datasets (SGD)", "Efficient"],
        cons: ["Sensitive to Learning Rate", "Can get stuck in local minima (non-convex)", "Requires feature scaling"]
    },
    codeSnippet: `import numpy as np
import matplotlib.pyplot as plt

# Cost function f(x) = x^2
x = np.linspace(-10, 10, 100)
y = x**2

# Gradient Descent
lr = 0.1
current_x = 8
path = [current_x]

for i in range(10):
    grad = 2 * current_x
    current_x = current_x - lr * grad
    path.append(current_x)

plt.plot(x, y)
plt.plot(path, np.array(path)**2, 'ro-')
plt.title("Gradient Descent on $x^2$")
plt.show()`,
    quiz: [
        { question: "What happens if the learning rate is too high?", options: ["Slow convergence", "Overshooting/Divergence", "Stuck in local minima", "Perfect convergence"], correctIndex: 1, explanation: "Large steps can cause the algorithm to bounce across the valley or diverge to infinity." }
    ],
    deepDive: {
        advancedTheory: "Momentum and Adam optimization add velocity terms to overcome local minima and saddle points.",
        keyFormulas: ["v_t = \\gamma v_{t-1} + \\eta \\nabla J(\\theta)", "\\theta = \\theta - v_t"],
        seminalPapers: []
    }
  },
  'tree-ensembles': {
    overview: `**Tree Ensembles** combine multiple Decision Trees to improve performance and stability. The two main families are **Bagging** (Random Forest) and **Boosting** (XGBoost, LightGBM, AdaBoost).`,
    mathematicalIntuition: `
**Random Forest (Bagging):**
Averages high-variance, low-bias trees trained on bootstrap samples.
$$ \\hat{f}(x) = \\frac{1}{B} \\sum_{b=1}^B f_b(x) $$

**Gradient Boosting:**
Trains trees sequentially, where each new tree $h_m(x)$ fits the **residuals** of the previous ensemble.
$$ F_{m}(x) = F_{m-1}(x) + \\eta h_m(x) $$
    `,
    useCases: ["Kaggle Competitions", "Churn Prediction", "Credit Risk Modeling"],
    prosCons: {
        pros: ["High Accuracy", "Handles non-linearities", "Feature Importance built-in"],
        cons: ["Slow inference (Forests)", "Hard to interpret (Black box)", "Boosting prone to noise"]
    },
    codeSnippet: `from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
clf = RandomForestClassifier(n_estimators=100).fit(X, y)

# Plot
xx, yy = np.mgrid[-2:3:.01, -2:2:.01]
probs = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
plt.contourf(xx, yy, probs, 25, cmap="RdBu", alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k')
plt.title("Random Forest Decision Boundary")
plt.show()`,
    quiz: [
        { question: "Why does Random Forest select a random subset of features at each split?", options: ["To reduce bias", "To decorrelate the trees", "To speed up training", "To increase variance"], correctIndex: 1, explanation: "Decorrelating trees reduces the variance of the ensemble average." }
    ],
    deepDive: {
        advancedTheory: "XGBoost uses a second-order Taylor expansion of the loss function for better convergence.",
        keyFormulas: ["Gain = \\frac{1}{2} [ \\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda} ] - \\gamma"],
        seminalPapers: []
    }
  },
  'backprop': {
    overview: `**Backpropagation** is the algorithm used to calculate the gradient of the loss function with respect to each weight in a neural network. It is essentially an efficient application of the chain rule.`,
    mathematicalIntuition: `
For a weight $w_{ij}^{(l)}$ connecting layer $l-1$ to $l$:
$$ \\frac{\\partial L}{\\partial w_{ij}^{(l)}} = \\frac{\\partial L}{\\partial a_j^{(l)}} \\cdot \\frac{\\partial a_j^{(l)}}{\\partial z_j^{(l)}} \\cdot \\frac{\\partial z_j^{(l)}}{\\partial w_{ij}^{(l)}} $$
where $\\delta_j^{(l)} = \\frac{\\partial L}{\\partial z_j^{(l)}}$ is the "error" term propagated backward.
    `,
    useCases: ["Deep Learning Training", "Automatic Differentiation systems (PyTorch/TensorFlow)"],
    prosCons: {
        pros: ["Computationally efficient", "Exact gradient calculation"],
        cons: ["Vanishing Gradients (deep nets)", "Requires differentiable activation functions"]
    },
    codeSnippet: `import numpy as np

# Simple Backprop for f(x) = x^2
x = 2.0
w = 3.0
b = 1.0

# Forward
z = w * x + b
a = z**2  # Activation/Loss

# Backward
dL_da = 1.0
da_dz = 2 * z
dz_dw = x

grad_w = dL_da * da_dz * dz_dw
print(f"Gradient w.r.t w: {grad_w}") # 2 * (3*2+1) * 2 = 28`,
    quiz: [
        { question: "What causes the Vanishing Gradient problem?", options: ["ReLU activation", "Sigmoid derivatives < 0.25", "High learning rates", "Too few layers"], correctIndex: 1, explanation: "Repeated multiplication of small derivatives (like sigmoid's max 0.25) drives gradients to zero in deep networks." }
    ],
    deepDive: {
        advancedTheory: "Backprop is a special case of Reverse Accumulation Mode in Automatic Differentiation.",
        keyFormulas: ["\\delta^{(l)} = ((\\mathbf{W}^{(l+1)})^T \\delta^{(l+1)}) \\odot \\sigma'(z^{(l)})"],
        seminalPapers: []
    }
  },
  'metrics': {
    overview: `**Evaluation Metrics** quantify the performance of ML models. The choice depends heavily on the business objective and class balance. Accuracy is often misleading.`,
    mathematicalIntuition: `
**Precision:** $TP / (TP + FP)$ (Quality of positive predictions)
**Recall:** $TP / (TP + FN)$ (Quantity of positives found)
**F1-Score:** Harmonic mean $2 \\cdot \\frac{P \\cdot R}{P + R}$
**ROC-AUC:** Area under the curve of TPR vs FPR at various thresholds.
    `,
    useCases: ["Fraud Detection (Recall priority)", "Spam Filter (Precision priority)", "Search Ranking (NDCG)"],
    prosCons: {
        pros: ["Standardized benchmarks", "Direct business alignment"],
        cons: ["Single number summaries lose nuance", "Goodhart's Law"]
    },
    codeSnippet: `from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()`,
    quiz: [
        { question: "In fraud detection where missing a fraudster is costly, which metric is most important?", options: ["Precision", "Recall", "Accuracy", "Specificty"], correctIndex: 1, explanation: "Recall minimizes False Negatives (missed fraud)." }
    ],
    deepDive: {
        advancedTheory: "Probabilistic interpretation: AUC is the probability that a random positive example is ranked higher than a random negative example.",
        keyFormulas: [],
        seminalPapers: []
    }
  },
  'regularization': {
    overview: `**Regularization** techniques prevent overfitting by adding a penalty term to the loss function, constraining the model complexity. Common types are L1 (Lasso) and L2 (Ridge).`,
    mathematicalIntuition: `
Loss with penalty:
$$ J(\\theta) = L(\\theta) + \\lambda R(\\theta) $$
**L1 (Lasso):** $R(\\theta) = \\sum |\\theta_i|$. Induces sparsity (feature selection).
**L2 (Ridge):** $R(\\theta) = \\sum \\theta_i^2$. Shrinks weights evenly.
**Elastic Net:** Combination of L1 and L2.
    `,
    useCases: ["High-dimensional data (Genomics)", "Preventing overfitting in small datasets", "Feature Selection"],
    prosCons: {
        pros: ["Improves generalization", "Lasso selects features", "Ridge handles multicollinearity"],
        cons: ["Introduces bias (Underfitting)", "Requires tuning lambda"]
    },
    codeSnippet: `from sklearn.linear_model import Ridge, Lasso
import numpy as np

# Ridge vs Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# L1 creates sparse weights (zeros)
# L2 keeps all weights small but non-zero
print("L1 leads to sparsity in high dimensions.")`,
    quiz: [
        { question: "Which regularization technique yields sparse models?", options: ["L1 (Lasso)", "L2 (Ridge)", "Dropout", "Batch Norm"], correctIndex: 0, explanation: "The L1 geometry (diamond shape) intersects cost contours at axes, forcing coefficients to zero." }
    ],
    deepDive: {
        advancedTheory: "Regularization is equivalent to Bayesian Inference with a prior on weights. L2 corresponds to a Gaussian Prior, L1 to a Laplacian Prior.",
        keyFormulas: ["P(\\theta|D) \\propto P(D|\\theta)P(\\theta)"],
        seminalPapers: []
    }
  },
  'pca': {
    overview: `**Principal Component Analysis (PCA)** is a dimensionality reduction technique that projects data onto orthogonal directions (Principal Components) of maximum variance.`,
    mathematicalIntuition: `
We seek vectors $\\mathbf{u}$ that maximize variance $\\text{Var}(\\mathbf{u}^T \\mathbf{X})$.
This reduces to finding the **eigenvectors** of the Covariance Matrix $\\Sigma = \\frac{1}{n}\\mathbf{X}^T\\mathbf{X}$.
$$ \\Sigma \\mathbf{v} = \\lambda \\mathbf{v} $$
The eigenvector with the largest eigenvalue $\\lambda$ is the first PC.
    `,
    useCases: ["Data Visualization (2D/3D)", "Noise Reduction", "Pre-processing for regression"],
    prosCons: {
        pros: ["Removes correlation", "Reduces storage/compute", "Unsupervised"],
        cons: ["Linear only (Kernel PCA for non-linear)", "Lower interpretability of components", "Loss of information"]
    },
    codeSnippet: `from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X = load_iris().data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=load_iris().target)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA on Iris')
plt.show()`,
    quiz: [
        { question: "What property do Principal Components have relative to each other?", options: ["Parallel", "Orthogonal", "Correlated", "Identical"], correctIndex: 1, explanation: "PCs are orthogonal (uncorrelated) by definition of the eigendecomposition of a symmetric matrix." }
    ],
    deepDive: {
        advancedTheory: "PCA minimizes the reconstruction error (Mean Squared Error) between original and projected data.",
        keyFormulas: ["\\mathbf{X} \\approx \\mathbf{T}\\mathbf{P}^T"],
        seminalPapers: []
    }
  },
  'norm-dist': {
    overview: `The **Normal Distribution** (Gaussian) is the most important probability distribution in statistics. It is symmetric, bell-shaped, and defined entirely by its mean $\\mu$ and variance $\\sigma^2$.`,
    mathematicalIntuition: `
PDF:
$$ f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2} $$
**Central Limit Theorem (CLT):** The sum of independent random variables tends toward a normal distribution, regardless of the original distribution.
    `,
    useCases: ["Hypothesis Testing (Z-test)", "Modeling errors/noise", "Process Control (Six Sigma)"],
    prosCons: {
        pros: ["Mathematically tractable", "CLT justification", "Defined by 2 parameters"],
        cons: ["Thin tails (underestimates extreme events)", "Not all data is normal"]
    },
    codeSnippet: `import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.fill_between(x, stats.norm.pdf(x, mu, sigma), alpha=0.2)
plt.title("Standard Normal Distribution")
plt.show()`,
    quiz: [
        { question: " approximately what percentage of data lies within 1 standard deviation of the mean?", options: ["50%", "68%", "95%", "99.7%"], correctIndex: 1, explanation: "The empirical rule (68-95-99.7)." }
    ],
    deepDive: {
        advancedTheory: "The Gaussian distribution has maximum entropy for a given variance.",
        keyFormulas: [],
        seminalPapers: []
    }
  },
  'sorting': {
    overview: `**Sorting Algorithms** rearrange elements in a specific order. The choice of algorithm depends on data size, memory constraints, and stability requirements.`,
    mathematicalIntuition: `
**Complexity Hierarchy:**
*   **O(N²):** Bubble, Insertion, Selection (Simple, slow for large N).
*   **O(N log N):** Merge, Quick, Heap (Standard for general purpose).
*   **O(N):** Radix, Counting (Non-comparison based).

**Master Theorem** for Merge Sort:
$$ T(n) = 2T(n/2) + O(n) \\implies O(n \\log n) $$
    `,
    useCases: ["Database Indexing (B-Trees)", "Search Preprocessing (Binary Search)", "Computer Graphics (Depth sorting)"],
    prosCons: {
        pros: ["Fundamental to CS", "Optimized libraries exist (Timsort)"],
        cons: ["Comparison sorts bounded by $N \\log N$"]
    },
    codeSnippet: `import random
import time

def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

data = [random.randint(0, 100) for _ in range(20)]
print(quicksort(data))`,
    quiz: [
        { question: "Which sorting algorithm is typically used for standard library implementations (like Python's sort)?", options: ["Quick Sort", "Merge Sort", "Timsort", "Bubble Sort"], correctIndex: 2, explanation: "Timsort is a hybrid stable sort (Merge + Insertion) optimized for real-world data." },
        { question: "What is the worst case time complexity of Quick Sort?", options: ["O(N)", "O(N log N)", "O(N^2)", "O(log N)"], correctIndex: 2, explanation: "O(N^2) occurs when the pivot selection is poor (e.g., already sorted array), though this is rare with randomized pivots." }
    ],
    deepDive: {
        advancedTheory: "Quick Sort is often faster in practice than Merge Sort due to better cache locality, despite having a worse worst-case time complexity.",
        keyFormulas: ["\\sum_{i=1}^N \\log i = \\log(N!) \\approx N \\log N"],
        seminalPapers: []
    }
  },
  'graph-algo': {
      overview: "**Graph Algorithms** solve problems related to networks, relationships, and pathfinding. Key structures are $G=(V, E)$ where $V$ are vertices and $E$ are edges.",
      mathematicalIntuition: `
**Dijkstra's Algorithm:**
Greedy approach to find shortest paths from source. Uses a priority queue.
$$ d(v) = \\min(d(v), d(u) + w(u, v)) $$

**PageRank:**
Eigenvector centrality measure.
$$ PR(u) = \\sum_{v \\in B_u} \\frac{PR(v)}{L(v)} $$
      `,
      useCases: ["Social Networks (Friend suggestions)", "Google Maps (Shortest Path)", "Network Routing (OSPF)"],
      prosCons: { pros: ["Modeling complex relationships", "Powerful abstractions"], cons: ["NP-Hard problems common (TSP)", "Scalability on massive graphs"] },
      codeSnippet: `import networkx as nx
import matplotlib.pyplot as plt

G = nx.erdos_renyi_graph(10, 0.3)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
print("Shortest path 0->4:", nx.shortest_path(G, 0, 4))
plt.show()`,
      quiz: [
          { question: "Which algorithm is used to traverse a graph layer by layer?", options: ["DFS", "BFS", "Dijkstra", "A*"], correctIndex: 1, explanation: "Breadth-First Search (BFS) explores neighbor nodes first before moving to the next level depth." }
      ],
      deepDive: { advancedTheory: "Graph Neural Networks (GNNs) generalize convolutions to graph structured data.", keyFormulas: [], seminalPapers: [] }
  },
  'dynamic-prog': {
      overview: "**Dynamic Programming (DP)** is an optimization technique that solves complex problems by breaking them down into simpler overlapping subproblems and storing the results (memoization).",
      mathematicalIntuition: `
**Bellman Equation:**
$$ V(s) = \\max_a (R(s,a) + \\gamma V(s')) $$
Key attributes:
1.  **Optimal Substructure:** Solution to problem contains solutions to subproblems.
2.  **Overlapping Subproblems:** Subproblems recur many times.
      `,
      useCases: ["Reinforcement Learning", "Sequence Alignment (DNA)", "Resource Allocation (Knapsack)"],
      prosCons: { pros: ["Reduces exponential time to polynomial", "Guarantees optimality"], cons: ["High space complexity (tables)", "Hard to formulate"] },
      codeSnippet: `def fib_dp(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print([fib_dp(n) for n in range(10)])`,
      quiz: [
           { question: "What distinguishes DP from Divide and Conquer?", options: ["Recursion", "Overlapping subproblems", "Base cases", "Complexity"], correctIndex: 1, explanation: "Divide and Conquer assumes subproblems are independent (like Merge Sort), while DP handles shared subdependencies." }
      ],
      deepDive: { advancedTheory: "The Knapsack problem is NP-Complete, but can be solved in pseudo-polynomial time using DP.", keyFormulas: [], seminalPapers: [] }
  },
  'deployment': {
      overview: "**Model Deployment** is the final stage of the ML lifecycle, bridging the gap between a trained artifact and a production service delivering value.",
      mathematicalIntuition: `
**Latency:** Total time for request $T_{total} = T_{net} + T_{infer} + T_{process}$.
**Throughput:** Requests per second (RPS).
Scaling often follows Little's Law: $L = \\lambda W$.
      `,
      useCases: ["Real-time fraud scoring", "Recommendation APIs", "Edge deployment (Mobile/IoT)"],
      prosCons: {
          pros: ["Realizes business value", "Scalable architectures (Kubernetes)"],
          cons: ["Training-Serving skew", "Monitoring complexity", "Latency constraints"]
      },
      codeSnippet: `# Pseudo-code for FastAPI
# from fastapi import FastAPI
# app = FastAPI()
# @app.post("/predict")
# async def predict(data: InputData):
#     result = model.predict(data.features)
#     return {"prediction": result}`,
      quiz: [
          { question: "What is a common format for serializing ML models?", options: ["CSV", "ONNX", "MP4", "HTML"], correctIndex: 1, explanation: "ONNX (Open Neural Network Exchange) allows models to be portable across frameworks." }
      ],
      deepDive: {
          advancedTheory: "Model Distillation trains a small 'student' network to mimic a large 'teacher' to reduce inference latency.",
          keyFormulas: [],
          seminalPapers: []
      }
  },
  'pandas-adv': {
      overview: "**Advanced Pandas** involves mastering vectorization, complex aggregation, and memory management for efficient data manipulation.",
      mathematicalIntuition: "Vectorization uses SIMD (Single Instruction, Multiple Data) operations at the CPU level, avoiding slow Python loops.",
      useCases: ["Feature Engineering pipelines", "Time-series resampling", "Complex ETL"],
      prosCons: { pros: ["High expressiveness", "Integration with ecosystem"], cons: ["Memory inefficient (copies data)", "Single-threaded"] },
      codeSnippet: `import pandas as pd
import numpy as np

df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'], 'B': np.random.randn(4)})
# GroupBy with Transform for window operations
df['B_mean'] = df.groupby('A')['B'].transform('mean')
print(df)`,
      quiz: [
          { question: "Which method is generally fastest for applying a function to a column?", options: [".apply(func)", "Vectorized operations", "for loop", "list comprehension"], correctIndex: 1, explanation: "Vectorized operations delegate to C-optimized NumPy routines." }
      ],
      deepDive: { advancedTheory: "Polars is a modern alternative to Pandas written in Rust that supports lazy evaluation and multi-threading.", keyFormulas: [], seminalPapers: [] }
  },
  'numpy-broadcasting': {
      overview: "**Broadcasting** describes how NumPy treats arrays with different shapes during arithmetic operations. It stretches smaller arrays to match larger ones without copying data.",
      mathematicalIntuition: `
Two dimensions are compatible when:
1. They are equal, or
2. One of them is 1.
If compatible, the array with size 1 is stretched.
      `,
      useCases: ["Normalizing data ($x - \\mu$)", "Image manipulation (RGB channels)", "Weighted averages"],
      prosCons: { pros: ["Memory efficient (no copies)", "Concise code"], cons: ["Implicit behavior can hide bugs", "Hard to debug shape mismatches"] },
      codeSnippet: `import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]]) # Shape (2, 3)
b = np.array([10, 20, 30])           # Shape (3,)
# b is broadcast to [[10, 20, 30], [10, 20, 30]]
print(A + b)`,
      quiz: [
          { question: "Can an array of shape (3, 1) be broadcast with an array of shape (3,)?", options: ["Yes, result (3, 3)", "Yes, result (3,)", "No", "Yes, result (3, 1)"], correctIndex: 0, explanation: "Yes. (3,1) and (1,3) (implicitly) broadcast to (3,3)." }
      ],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'clean-code': {
      overview: "**Clean Code** principles are vital for Data Science. Notebooks often breed technical debt; moving to modular, tested code is key for production.",
      mathematicalIntuition: "N/A",
      useCases: ["Code Reviews", "CI/CD Pipelines", "Library development"],
      prosCons: { pros: ["Maintainability", "Reduced bugs", "Easier collaboration"], cons: ["Higher upfront effort", "Learning curve"] },
      codeSnippet: `from typing import List

def calculate_std_dev(values: List[float]) -> float:
    """
    Calculates population standard deviation.
    Raises ValueError if list is empty.
    """
    if not values:
        raise ValueError("Values cannot be empty")
    mu = sum(values) / len(values)
    return (sum((x - mu)**2 for x in values) / len(values)) ** 0.5`,
      quiz: [
          { question: "What is 'Type Hinting' in Python?", options: ["Enforced static typing", "Annotations for IDEs/Linters", "A deprecated feature", "Faster execution"], correctIndex: 1, explanation: "Type hints help tools verify correctness but do not change runtime behavior." }
      ],
      deepDive: { advancedTheory: "SOLID Principles (Single responsibility, Open-closed, Liskov substitution, Interface segregation, Dependency inversion).", keyFormulas: [], seminalPapers: [] }
  },
  'generators': {
      overview: "**Generators** allow you to declare a function that behaves like an iterator. They yield items one at a time, enabling processing of data larger than memory.",
      mathematicalIntuition: "Lazy Evaluation.",
      useCases: ["Streaming large logs", "Infinite data pipelines", "Pipelining processing steps"],
      prosCons: { pros: ["Low memory footprint", "Composable"], cons: ["One-time iteration", "No random access"] },
      codeSnippet: `def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

gen = infinite_sequence()
print(next(gen)) # 0
print(next(gen)) # 1`,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'pipelines': {
      overview: "**Data Pipelines** automate the flow of data from source to destination, transforming it along the way. Tools like Airflow, Prefect, or Dagster manage these workflows.",
      mathematicalIntuition: "Modeled as Directed Acyclic Graphs (DAGs) where nodes are tasks and edges are dependencies.",
      useCases: ["Daily ETL jobs", "Model retraining workflows", "Data warehousing"],
      prosCons: { pros: ["Reproducibility", "Error handling/Retries", "Scheduling"], cons: ["Infrastructure complexity", "Debuggability"] },
      codeSnippet: `# Airflow Concept
# with DAG('my_dag') as dag:
#    t1 = BashOperator(task_id='print_date', bash_command='date')
#    t2 = PythonOperator(task_id='train', python_callable=train_model)
#    t1 >> t2`,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'monitoring': {
      overview: "**Monitoring & Drift Detection** involves tracking model performance in production to ensure it hasn't degraded due to changing data distributions.",
      mathematicalIntuition: `
**Kullback-Leibler (KL) Divergence:**
$$ D_{KL}(P || Q) = \\sum P(x) \\log \\frac{P(x)}{Q(x)} $$
Used to measure distance between training ($P$) and serving ($Q$) distributions.
      `,
      useCases: ["Detecting Concept Drift", "Detecting Data Drift", "Outlier detection"],
      prosCons: { pros: ["Proactive maintenance", "Trust"], cons: ["False alarms", "Choosing right metric is hard"] },
      codeSnippet: `from scipy.spatial.distance import jensenshannon
import numpy as np

p = np.array([0.1, 0.4, 0.5])
q = np.array([0.2, 0.3, 0.5])
drift_score = jensenshannon(p, q)
print(f"Drift Score: {drift_score:.4f}")`,
      quiz: [
          { question: "What is 'Concept Drift'?", options: ["Input data changes", "The relationship between input and target changes", "Model weights change", "Code bugs"], correctIndex: 1, explanation: "Concept drift means the mapping X->Y has changed (e.g., spam definitions change over time)." }
      ],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'svm': {
      overview: "**Support Vector Machines (SVM)** find the optimal hyperplane that maximizes the margin between classes.",
      mathematicalIntuition: "Maximize $\\frac{2}{\\|w\\|}$ subject to $y_i(w^Tx_i + b) \\ge 1$.",
      useCases: ["Text classification", "Bioinformatics"],
      prosCons: { pros: ["Global optimum", "Kernel trick"], cons: ["Slow for large N", "No probability outputs"] },
      codeSnippet: `from sklearn.svm import SVC
clf = SVC(kernel='linear').fit(X, y)`,
      quiz: [],
      deepDive: { advancedTheory: "Mercer's Theorem.", keyFormulas: [], seminalPapers: [] }
  },
  'cv-split': {
      overview: "**Cross-Validation** estimates generalization error by splitting data into K folds.",
      mathematicalIntuition: "$E \\approx \\frac{1}{K} \\sum E_k$",
      useCases: ["Model selection", "Hyperparameter tuning"],
      prosCons: { pros: ["Robust estimate"], cons: ["Computationally expensive"] },
      codeSnippet: `from sklearn.model_selection import cross_val_score`,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'bayes-inf': {
      overview: "**Bayesian Inference** updates the probability for a hypothesis as more evidence becomes available.",
      mathematicalIntuition: "$P(H|E) = \\frac{P(E|H)P(H)}{P(E)}$",
      useCases: ["A/B Testing", "Parameter estimation"],
      prosCons: { pros: ["Incorporates priors"], cons: ["Computational cost"] },
      codeSnippet: ``,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'p-value': {
      overview: "**Hypothesis Testing** uses p-values to determine if results are statistically significant.",
      mathematicalIntuition: "$P(Data | H_0)$",
      useCases: ["Clinical trials", "A/B testing"],
      prosCons: { pros: ["Standard"], cons: ["Misinterpreted"] },
      codeSnippet: `from scipy import stats
t, p = stats.ttest_ind([1,2], [5,6])`,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'bias-variance': {
      overview: "The **Bias-Variance Tradeoff** is the conflict between minimizing bias (underfitting) and variance (overfitting).",
      mathematicalIntuition: "$MSE = Bias^2 + Variance + \\text{Noise}$",
      useCases: ["Model selection"],
      prosCons: { pros: ["Fundamental"], cons: ["Abstract"] },
      codeSnippet: ``,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'distributions': {
      overview: "Common **Probability Distributions** model different types of data generation processes.",
      mathematicalIntuition: "Poisson (Counts), Bernoulli (Binary), Exponential (Wait times).",
      useCases: ["Risk modeling"],
      prosCons: { pros: [], cons: [] },
      codeSnippet: ``,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  },
  'feature-eng': {
      overview: "**Feature Engineering** is the process of using domain knowledge to extract features from raw data.",
      mathematicalIntuition: "Mapping $x \\to \\phi(x)$.",
      useCases: ["Kaggle", "Time-series"],
      prosCons: { pros: ["Higher accuracy"], cons: ["Time consuming"] },
      codeSnippet: `df['log_x'] = np.log(df['x'])`,
      quiz: [],
      deepDive: { advancedTheory: "", keyFormulas: [], seminalPapers: [] }
  }
};