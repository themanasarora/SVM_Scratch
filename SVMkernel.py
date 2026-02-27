import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01,n_iter=1000, kernel="linear", gamma=1):
        
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iter
        
        self.kernel_type = kernel
        self.gamma = gamma
        
        self.alphas = None
        self.b = 0
        
        self.X = None
        self.y = None

    # ---------------------------
    # Kernel Functions
    # ---------------------------
    def kernel(self, x1, x2):
        if self.kernel_type == "linear":
            return np.dot(x1, x2)
        
        elif self.kernel_type == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

        else:
            raise ValueError("Unsupported kernel")

    # ---------------------------
    # Training
    # ---------------------------
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples = X.shape[0]
        
        self.alphas = np.zeros(n_samples)
        self.X = X
        self.y = y_
        self.b = 0
        
        C = 1 / self.lambda_param   # Soft-margin control
        
        for _ in range(self.n_iters):
            for i in range(n_samples):
                
                # Compute prediction for x_i
                prediction = 0
                for j in range(n_samples):
                    prediction += (
                        self.alphas[j] *
                        self.y[j] *
                        self.kernel(self.X[j], self.X[i])
                    )
                
                # Hinge loss condition
                if y_[i] * (prediction - self.b) < 1:
                    self.alphas[i] += self.lr
                    
                    # Clip alphas (soft margin constraint)
                    self.alphas[i] = np.clip(self.alphas[i], 0, C)
                    
                    # Update bias
                    self.b -= self.lr * y_[i]

    # ---------------------------
    # Prediction
    # ---------------------------
    def predict(self, X):
        y_pred = []
        
        for x in X:
            prediction = 0
            for alpha, y_i, x_i in zip(self.alphas, self.y, self.X):
                prediction += alpha * y_i * self.kernel(x_i, x)
            
            y_pred.append(np.sign(prediction - self.b))
        
        return np.array(y_pred)