import numpy as np

class MF:
    def __init__(self, k, epochs, lr):
        """
        Initialize Matrix Factorization model.
        
        Args:
            k (int): Number of latent factors.
            epochs (int): Number of training iterations.
            lr (float): Learning rate for gradient descent.
        """
        self.k = k
        self.epochs = epochs
        self.lr = lr
        self.users = None
        self.items = None
        self.n_users = None
        self.n_items = None
        self.users_map = None
        self.items_map = None
        self.U = None
        self.I = None

    def fit(self, R):
        """
        Fit the model to the training data.
        
        Args:
            R : user-item rating matrix.

        Returns:
            List : loss_history
        """
        self.users = set()
        self.items = set()
        for user_id, item_id, *_ in R:
            self.users.add(user_id)
            self.items.add(item_id)
        self.users = sorted(self.users)
        self.items = sorted(self.items)
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        
        self.users_map = {user_id : idx for idx, user_id in enumerate(self.users)}
        self.items_map = {item_id : idx for idx, item_id in enumerate(self.items)}

        U = np.random.normal(0, 0.1, (self.n_users, self.k))
        I = np.random.normal(0, 0.1, (self.n_items, self.k))

        #training loop
        observed_ratings = len(R)
        loss_history = []
        for epoch in range(self.epochs):
            loss = 0
            np.random.shuffle(R)
            for instance in R:
                i,j = self.users_map[int(instance[0])], self.items_map[int(instance[1])]

                # gradient descent
                rating = instance[2]
                error = rating - np.dot(U[i], I[j])
                U_i = U[i].copy()
                U[i] += self.lr*error*I[j]
                I[j] += self.lr*error*U_i

                loss += error**2

            loss /= observed_ratings
            loss_history.append(loss)
            print(f"{epoch=}\n{loss=:.4f}")
        self.U = U
        self.I = I
        return loss_history

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID.
            item_id: Item ID.
            
        Returns:
            float: Predicted rating.
        """
        i, j = self.users_map[user_id], self.items_map[item_id]
        return np.dot(self.U[i], self.I[j])

    def recommend(self, user_id, k):
        """
        Recommend top k items for a user.
        
        Args:
            user: User ID.
            k (int): Number of items to recommend.
            
        Returns:
            list: List of (item_id, predicted_rating) tuples, sorted by rating.
        """
        recommendations = []
        user = self.users_map[user_id]
        for item in self.items:
            recommendations.append(np.dot(self.U[user], self.I[item]))
        recommendations.sort(reverse=True)
        return recommendations[:k]
    def evaluate(self, R):
        """
        Evaluate the model on test data using RMSE.
        
        Args:
            R : user-item rating matrix for validation.
            
        Returns:
            float: Root Mean Squared Error (RMSE).
        """
        mse = 0
        n_samples = len(R)
        for user_id, item_id, rating, *_ in R:
            error = rating - self.predict(user_id, item_id)
            mse += error**2
        rmse = np.sqrt(mse / n_samples)
        return rmse
    
if __name__ == "__main__":
    pass