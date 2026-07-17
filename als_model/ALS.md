# Alternating Least Squares - ALS
> Explain the flow ALS algorithm and its hyperparameters. Code is in `als_model/als.py` file.

## Introduction

Alternating Least Squares is a matrix factorization technique used in collaborative filtering for recommendation systems.The goal of the algorithm is to predict missing values ​​in a User-Item interaction matrix - a rating matrix.

ALS approximates the rating matrix $R$ ($M \times N$) as the product of two lower-dimension matrices: 

- User feature matrix $U$ ($M \times K$), $K$ is the number of latent factors.
- Item feature matrix $V$ ($N \times K$), $K$ is the number of latent factors.

**Goal:** find $U$ and $V$ such that $ R \approx U \cdot V^T $.

## Hyperparameters

- `n_factors`: Number of latent factors - $K$. This is the dimensionality of the vector space for user features and item features. A higher value results in more detailed model but increases the risk of overfitting and slows down computation.

- `regularization`: L2 regularization coefficient - $\lambda$. This parameter helps control the size of the weights in matrices $U$ and $V$. Preventing the vectors from reaching excessively large values that would lead to overfitting.

- `iterations`: Number of iterations for the ALS algorithm. Each iteration consists of two steps: fixing $V$ and solving for $U$, then fixing $U$ and solving for $V$. More iterations can lead to better convergence but increase computation time.

- `user_factors`: Matrix $U$ has shape $(n\_users, n\_factors)$ - User features.

- `item_factors`: Matrix $V$ has shape $(n\_items, n\_factors)$ - Item features.

## Algorithm Flow

### `fit(self, R)`

**Step 1: Random Initialization**

- The model intializes the user_factors $U$ and item_factors $V$ matrices with very small random values following a Gaussian distribution.

- It pre-computes the regularization component: `lambda_I = self.regularization * np.eye(self.n_factors)`. This generates an identity matrix $\lambda I_K$ which is later added to the linear system to ensure stability.

**Step 2: The Alternating Optimization Process**

In loop `for step in range(self.iterations)`:

**Phase 1:** Fix item_factors $V$ and update user_factors $U$

- For each user $u$, the algorithm identifies the features of the items they have actually rated (Where $R[u, :] > 0$). It extracts a subset of the item matrix (V_u) corresponding only to these observed items.
 
**Phase 2:** Fix user_factors $U$ and update item_factors $V$


## References

1. [CME 323 - Lecture 14](https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf)