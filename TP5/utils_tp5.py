import numpy as np

def generate_regression_task(n_data = 500, n_test = 200):
    """
    Construit un jeu de données de régression avec n_data observations,
    Les observations sont en dimension 1
    Les labels sont en dimension 1
    
    Renvoie x_train, y_train, x_test, y_test --> (observations, labels) pour train et test. 
    """
    x_train = np.random.uniform(0, 4, size=n_data)
    y_train = np.sin(x_train**2) + x_train + 0.1 * np.random.randn(n_data)
    x_test = np.random.uniform(0, 4, size=n_test)
    y_test = np.sin(x_test**2) + x_test + 0.1 * np.random.randn(n_test)
    return x_train, y_train, x_test, y_test


def generate_classification_task(n_data = 1000, n_test = 1000):
    """
    construit un jeu de données de classification avec n_data observations et 3 classes
    Les observations sont en dimension 2
    Les labels vallent 0, 1, ou 2
    
    Renvoie x_train, y_train, x_test, y_test --> (observations, labels) pour train et test. 
    """
    n1 = n_data//3
    n2 = n_data//3
    n3 = n_data - (n1 + n2)
    
    x1 = np.random.randn(n1, 2)
    x2 = np.random.randn(n2, 2)
    x2[:,1] = x2[:,1]**2 + 3
    x3 = np.random.randn(n3, 2)
    x3[:,1] = -x3[:,1]**2 - 3 + x3[:,0]**2
    
    x_train = np.concatenate([x1, x2, x3], axis=0)
    
    y_train = np.array([0]*n1 + [1]*n2 + [2]*n3)
    
    # Et rebelotte pour le test set. Moche mais bon...
    
    n1 = n_test//3
    n2 = n_test//3
    n3 = n_test - (n1 + n2)
    
    x1 = np.random.randn(n1, 2)
    x2 = np.random.randn(n2, 2)
    x2[:,1] = x2[:,1]**2 + 3
    x3 = np.random.randn(n3, 2)
    x3[:,1] = -x3[:,1]**2 - 3 + x3[:,0]**2
    
    x_test = np.concatenate([x1, x2, x3], axis=0)
    
    y_test = np.array([0]*n1 + [1]*n2 + [2]*n3)
    
    return x_train, y_train, x_test, y_test
