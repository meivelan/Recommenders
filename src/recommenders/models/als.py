import numpy as np
from recommenders.evaluation.regression_metrics import rmse

class ALS:
    """
    Alternating Least Squares (ALS) Implementation.
    """
    def __init__(self, k=10, epochs=100, lr=0.01):
        """
        Initialize ALS model.
        
        Parameters:
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
        self.users_map_reverse = None
        self.items_map_reverse = None
        self.U = None
        self.I = None

    def fit(self, R):
        self.users = set()
        self.items = set()
        for user_id, item_id, rating, *_ in R:
            self.users.add(user_id)
            self.items.add(item_id)

        self.users = sorted(self.users)
        self.items = sorted(self.items)
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        
        
        self.users_map = {user_id : idx for idx, user_id in enumerate(self.users)}
        self.items_map = {item_id : idx for idx, item_id in enumerate(self.items)}

        self.users_map_reverse = {idx : user_id for idx, user_id in enumerate(self.users)}
        self.items_map_reverse = {idx : item_id for idx, item_id in enumerate(self.items)}


        U = np.random.normal(0, 0.1, (self.n_users, self.k))
        I = np.random.normal(0, 0.1, (self.n_items, self.k))

        loss_history = []

        for epoch in range(self.epochs):
            print(epoch)
            np.random.shuffle(R)

            for idx, user in enumerate(U):
                user_id = self.users_map_reverse[idx]
                rated_items = [(iid, rating) for uid, iid, rating, *_ in R if uid == user_id]
                for item_id, rating in rated_items:
                    item_idx = self.items_map[item_id]
                    error = rating - np.dot(user, I[item_idx])
                    user += self.lr*error*I[item_idx]

            for idx, item in enumerate(I):
                item_id = self.items_map_reverse[idx]
                rated_users = [(uid, rating) for uid, iid, rating, *_ in R if iid == item_id]
                for user_id, rating in rated_users:
                    user_idx = self.users_map[user_id]
                    error = rating - np.dot(U[user_idx], item)
                    item += self.lr*error*U[user_idx]

            loss = 0
            for user_id, item_id, rating, *_ in R:
                i, j = self.users_map[user_id], self.items_map[item_id]
                error = rating - np.dot(U[i], I[j])
                loss += error**2
            
            loss /= len(R)
            loss_history.append(loss)
            print(f"{epoch=} - {loss=}")
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
        user_id = self.users_map[user_id]
        for item_id in self.items:
            recommendations.append((user_id, item_id, np.dot(self.U[user_id], self.I[item_id])))
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:k]

    def evaluate(self, R):
        mse = 0
        n_samples = len(R)
        for user_id, item_id, rating, *_ in R:
            error = rating - self.predict(user_id, item_id)
            mse += error**2
        rmse = np.sqrt(mse / n_samples)
        return rmse


if __name__=="__main__":
    pass
