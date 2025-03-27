import numpy as np

class LassoHomotopy:
    def __init__(self, alpha=1.0, tol=1e-4, max_iter=1000, verbose=False, 
                 warm_start=False, adaptive_alpha=False, normalize=False, 
                 random_state=None, patience=10):
        """
        Initialize the LassoHomotopy model.

        Parameters:
        - alpha: Regularization parameter.
        - tol: Tolerance for convergence.
        - max_iter: Maximum number of iterations.
        - verbose: If True, prints iteration details.
        - warm_start: If True, reuses previous coefficients for faster convergence.
        - adaptive_alpha: If True, dynamically adjusts alpha based on feature importance.
        - normalize: If True, normalizes X before standardization.
        - random_state: Seed for reproducibility.
        - patience: Number of iterations without improvement before stopping.
        """
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.warm_start = warm_start
        self.adaptive_alpha = adaptive_alpha
        self.normalize = normalize
        self.random_state = random_state
        self.patience = patience

        # Model attributes
        self.coef_ = None
        self.intercept_ = None
        self.X_mean_ = None
        self.X_std_ = None
        self.y_mean_ = None

    def fit(self, X, y):
        """
        Fit the LASSO model using the homotopy method.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,)

        Returns:
        - self: The fitted model.
        """
        np.random.seed(self.random_state)
        n, p = X.shape

        # Normalize features if required
        if self.normalize:
            norms = np.linalg.norm(X, axis=0) + 1e-8
            X = X / norms

        # Standardization
        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0) + 1e-8
        X_std = (X - self.X_mean_) / self.X_std_
        
        self.y_mean_ = np.mean(y)
        y_centered = y - self.y_mean_

        # Initialize coefficients (warm start if enabled)
        if self.warm_start and self.coef_ is not None:
            coef = self.coef_
        else:
            coef = np.zeros(p)

        intercept = self.y_mean_

        # Compute feature correlations
        correlation = np.dot(X_std.T, y_centered) / n
        active_set = np.where(np.abs(correlation) > 0)[0]

        best_loss = np.inf
        no_improvement_count = 0

        for iteration in range(self.max_iter):
            coef_old = coef.copy()

            # Update coefficients using coordinate descent
            for j in active_set:
                # Compute the partial residual (adjusting for the current coefficient)
                rho_j = np.dot(X_std[:, j], y_centered - np.dot(X_std, coef) + coef[j] * X_std[:, j])
                gradient = np.dot(X_std[:, j], X_std[:, j])
                
                # Adaptive alpha adjustment (if enabled)
                alpha_j = self.alpha if not self.adaptive_alpha else self.alpha / (np.abs(rho_j) + 1e-6)
                
                # Corrected soft-thresholding update:
                coef[j] = np.sign(rho_j) * max(0, np.abs(rho_j) - alpha_j) / gradient

            # Check for convergence using relative change
            delta = np.linalg.norm(coef - coef_old, ord=2) / (np.linalg.norm(coef_old, ord=2) + 1e-8)
            loss = np.sum((y_centered - np.dot(X_std, coef)) ** 2) / (2 * n)

            if delta < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at iteration {iteration}")
                    break

            # Update active set dynamically
            active_set = np.where(np.abs(np.dot(X_std.T, y_centered - np.dot(X_std, coef))) > self.alpha)[0]

        self.coef_ = coef
        self.intercept_ = self.y_mean_
        return self


    def predict(self, X):
        """
        Predict using the fitted Lasso model.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)

        Returns:
        - predictions: numpy array of shape (n_samples,)
        """
        if self.normalize:
            norms = np.linalg.norm(X, axis=0) + 1e-8
            X = X / norms

        X_std = (X - self.X_mean_) / self.X_std_
        return np.dot(X_std, self.coef_) + self.intercept_
