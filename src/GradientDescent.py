import numpy as np
import matplotlib.pyplot as plt 

def MSE(y,yp):
    return np.sum((y- yp)**2)/len(y)


def gradient_descent(x,y,iterations = 1000, lr = 0.01, threshold = 0.0001):
    n = len(x)
    w = np.random.random()
    bias = 0.01

    previous_cost = None 

    for i in range(iterations):
        yp = w*x + bias 
        cost = MSE(y,yp)

        if previous_cost and abs(previous_cost-cost)<=threshold:
            break

        previous_cost = cost 

        dw = -(2/n) * np.sum(x * (y-yp))
        dbias = -(2/n) * np.sum((y-yp))

        w -= lr*dw 
        bias -= lr*dbias 

    print(f"stopped after {i} iterations")
    
    plt.scatter(x,y)
    plt.plot(x,yp)
    plt.show()


n = 100
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)
gradient_descent(x,y)

