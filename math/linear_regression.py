
import numpy as np
import matplotlib.pyplot as mplot

def plotter(x:list, y:list, intercept:float, slope:float):
    
    mplot.plot(x,y, "bo", x, np.array(x) * slope + intercept, '-k' )
    mplot.show()


# performs simple linear regression
def linear_regression(weights: list, visual: bool = None) -> np.ndarray:
    # weights represents y values of the regression, boolean is for seeing a plot

    dim = len(weights)

    y = np.array(weights)

    x_array = []
    pre_X = [None] * dim # make shallow copies the size of the matrix dimensions
    for i in range(dim):
        pre_X[i] = [1, i/dim] # make design matrix
        x_array += [i/dim] # register plotting domain
    
    X = np.array(pre_X) 
    β = np.linalg.lstsq(X, y) # linear regression

    if visual: # debugging and plotting
        plotter(x_array, y, β[0][0], β[0][1])    
    return β


def test_primal(): 
    print(linear_regression([0.2,0.4,0.8,0.6,0.75,0.40,0.02,0.015], True))
