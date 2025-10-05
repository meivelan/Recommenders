import numpy as np
import pandas as pd
from recommenders.evaluation.regression_metrics import rmse,mse,mae


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
            print(f"{epoch=}: {loss=:.4f}")
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
        i, j = self.users_map.get(user_id, None), self.items_map.get(item_id, None)
        if not i or not j:
            return 0
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
        user_id = self.users_map[user_id]
        for item_id in self.items:
            rating_pred = np.dot(self.U[user_id], self.I[item_id])
            recommendations.append((item_id, rating_pred))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:k]
    
    def evaluate(self, R):
        """
        Evaluate the model on test data using RMSE.
        
        Args:
            R : user-item rating matrix for validation.
            
        Returns:
            float: Root Mean Squared Error (RMSE).
        """
        rating_true = np.empty(len(R))
        rating_pred = np.empty(len(R))
        for idx, (user_id, item_id, rating, *_) in enumerate(R):
            rating_true[idx] = rating
            rating_pred[idx] = self.predict(int(user_id), int(item_id))

        results = pd.DataFrame(
            {
                "rmse" : rmse(rating_true, rating_pred),
                "mse"  : mse(rating_true, rating_pred),
                "mae"  : mae(rating_true, rating_pred)
            },
            index = ["MF"]
        )
        return results
if __name__ == "__main__":
    pass