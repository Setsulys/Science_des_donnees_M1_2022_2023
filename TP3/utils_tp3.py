import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d


import sklearn
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing


'''
Author: Théo Lacombe, Jérémie Decock
Note: part of this file is taken from a TP that was prepared when I was Ph.D. Student at École polytechnique, in collaboration with Jérémie Decock and J.-B. Scoggins.
'''


def gen_1d_linear_regression_samples(a = 2, b = 0, n_samples = 20, s=2):

    x = np.random.uniform(low=-10., high=10., size=n_samples)
    y = a * x + b + np.random.normal(scale=s, size=x.shape)

    df = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])

    df = sklearn.utils.shuffle(df).reset_index(drop=True)
    
    return df


def gen_1d_polynomial_regression_samples(n_samples = 15):

    x = np.random.uniform(low=0., high=10., size=n_samples)
    #x = np.random.uniform(low=0., high=1., size=n_samples)

    y = 3. - 2. * x + x ** 2 - x ** 3 + np.random.normal(scale=10., size=x.shape)
    #y = np.cos(1.5 * np.pi * x) + np.random.normal(scale=0.1, size=x.shape)

    df = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])

    df = sklearn.utils.shuffle(df).reset_index(drop=True)

    return df


def plot_1d_regression_samples(dataframe, model=None, theta=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    df = dataframe  # make an alias
    
    ERROR_MSG1 = "The `dataframe` parameter should be a Pandas DataFrame having the following columns: ['x', 'y']"
    assert df.columns.values.tolist() == ['x', 'y'], ERROR_MSG1
    
    if model is not None:
        
        # Compute the model's prediction
        
        x_pred = np.linspace(df.x.min(), df.x.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_pred)
        
        df_pred = pd.DataFrame(np.array([x_pred.flatten(), y_pred.flatten()]).T, columns=['x', 'y'])
        
        df_pred.plot(x='x', y='y', style='r--', ax=ax)

    if theta is not None:
        if type(theta)==float:
            x_pred = np.linspace(df.x.min(), df.x.max(), 100).reshape(-1, 1)
            y_pred = x_pred * theta
            df_pred = pd.DataFrame(np.array([x_pred.flatten(), y_pred.flatten()]).T, columns=['x', 'y'])
        
            df_pred.plot(x='x', y='y', style='r--', ax=ax)
            
        elif type(theta)==np.ndarray:
            x_pred = np.linspace(df.x.min(), df.x.max(), 100).reshape(-1, 1)
            y_pred = x_pred * theta[0] + theta[1]
            df_pred = pd.DataFrame(np.array([x_pred.flatten(), y_pred.flatten()]).T, columns=['x', 'y'])
        
            df_pred.plot(x='x', y='y', style='r--', ax=ax)

    # Plot also the training points
    
    df.plot.scatter(x='x', y='y', ax=ax, grid=True)
    
    delta_y = df.y.max() - df.y.min()
    
    plt.ylim((df.y.min() - 0.15 * delta_y,
              df.y.max() + 0.15 * delta_y))

    
def gen_nd_dataset(dim=2, n_samples=50, scale = 2., A=None, b=None):
    x = np.random.uniform(low=-10., high=10., size=(n_samples, dim))
    if A is None:
        A = np.random.uniform(low = -1, high = 1, size=(dim, 1))
    if b is None:
        b = np.random.uniform(low = -1, high = 1)
    y = x.dot(A) + b + np.random.normal(scale = scale)
    
    return x, y


def plot_2d_with_color(X, y):
    fig, ax = plt.subplots()
    z = ax.scatter(X[:,0], X[:,1], c=y)
    ax.grid()
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    cbar = plt.colorbar(z)
    cbar.set_label('y')
    
    
def gen_log_data(theta = 2.489, n_samples = 100):
    x = np.random.uniform(low = -7, high = 7, size=n_samples)
    y = np.log( 1 + np.exp(theta * x)) + np.random.normal(scale=2, size=x.shape)
    
    return x, y


def plot_1d_log_model(x, y, theta=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    if theta is not None:
        ts = np.linspace(-7, 7, 100)
        ax.plot(ts, np.log(1 + np.exp(theta * ts)), c='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()

    
def plot_gd_log_model(list_thetas, list_losses, x, y, loss):
    fig, ax = plt.subplots()
    thetas = np.linspace(-3, 8, 100)
    ax.plot(thetas, [loss(theta, x, y) for theta in thetas], linestyle='dashed')
    
    ax.scatter(list_thetas, list_losses, marker='o', c='green', s=100)

    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$L(\theta)$')
    ax.grid()
    