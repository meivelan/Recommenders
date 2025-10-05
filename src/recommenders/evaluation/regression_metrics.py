import numpy as np

def rmse(rating_true, rating_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted ratings.
    
    Parameters:
        rating_true : Array of true ratings.
        rating_pred : Array of predicted ratings.
    
    Returns:
        float: The RMSE value.
    """
    return np.sqrt(np.mean(np.pow((rating_true - rating_pred), 2)))

def mse(rating_true, rating_pred):
    """
    Calculate the Mean Squared Error (MSE) between true and predicted ratings.
    
    Parameters:
        rating_true : Array of true ratings.
        rating_pred : Array of predicted ratings.
    
    Returns:
        float: The MSE value.
    """
    return np.mean(np.pow((rating_true - rating_pred), 2))

def mae(rating_true, rating_pred):
    """
    Calculate the Mean Absolute Error (MAE) between true and predicted ratings.
    
    Parameters:
        rating_true : Array of true ratings.
        rating_pred : Array of predicted ratings.
    
    Returns:
        float: The MAE value.
    """
    return np.mean(np.abs(rating_true - rating_pred))