import numpy as np
import matplotlib.pyplot as plt 
from autograd import grad 

def MSE(y,yp):
    return np.sum((y- yp)**2)/len(y)

#Simple Gradient descent with momentum
def gradient_descent(x,y,iterations = 1000, lr = 0.01, threshold = 0.000001, momentum=0.1):
    n = len(x)
    w = np.random.random()
    bias = 0.01

    previous_cost = None 
    change_w = 0
    change_bias = 0
    for i in range(iterations):
        yp = w*x + bias 
        cost = MSE(y,yp)

        if previous_cost and abs(previous_cost-cost)<=threshold:
            break

        previous_cost = cost 

        dw = -(2/n) * np.sum(x * (y-yp))
        dbias = -(2/n) * np.sum((y-yp))

        change_w = lr*dw + momentum*change_w
        change_bias = lr*dbias + momentum*change_bias

        w -= change_w
        bias -= change_bias

    print(f"stopped after {i} iterations")
    print("MSE: ",MSE(y,yp))
    
    plt.scatter(x,y)
    plt.plot(x,yp)
    plt.show()


n = 100
np.random.seed(4)
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)

#gradient_descent(x,y, lr=0.35, momentum=0.5)
"""
Stops after 27 iterations, which seems to be the best for this certain data set.
This is for the specific seed, and could vary
"""

#Method for a varying step length/learning rate
def step_length(t,t0,t1):
    return t0/(t+t1)


def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)


#training_gradient = grad(CostOLS,2)

def StocastichGD(x,y,iterations = 1000, t0 = 30, t1=10, threshold = 0.000001, momentum=0.1, M=5):
    n = len(x)
    w = np.random.random()
    bias = 0.01
    m = int(n/M) #number of minibatches
    previous_cost = None 
    change_w = 0
    change_bias = 0

    stop_loop = False
    num_iters = iterations
    for i in range(iterations):
        if stop_loop:
            break
        for j in range(m):
            k = np.random.randint(m)
            idx_low = int(n/m * k)
            idx_high = int(n/m * (k+1))
            
            new_x = x[idx_low:idx_high]
            new_y = y[idx_low:idx_high]

            yp = w*new_x + bias 

            cost = MSE(new_y,yp)*m

            if previous_cost and abs(previous_cost-cost)<=threshold:
                num_iters = i
                stop_loop = True
                break

            previous_cost = cost 

            dw = -(2/n) * np.sum(new_x * (new_y-yp))
            dbias = -(2/n) * np.sum((new_y-yp))

            t = i*m + j 
            lr = step_length(t,t0,t1)
            change_w = lr*dw + momentum*change_w
            change_bias = lr*dbias + momentum*change_bias

            w -= change_w
            bias -= change_bias

    print(f"stopped after {num_iters} iterations")
    yp = w*x + bias 
    print("MSE: ",MSE(y,yp))

    plt.scatter(x,y)
    plt.plot(x,yp)
    plt.show()


n = 100
np.random.seed(4) #Place seed here aswell for more accurate results
x = 2*np.random.rand(n,1)
y = 4+3*x+np.random.randn(n,1)
StocastichGD(x,y,momentum=0.5)







# objective function
def objective(x):
    return x**2.0 
 
# derivative of objective function
def derivative(x):
    return x * 2.0 

# gradient descent algorithm with momentum
def gradient_descentMomentum(objective, derivative, bounds, n_iter, step_size, momentum):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial point
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # keep track of the change
    change = 0.0
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution)
        # calculate update
        new_change = step_size * gradient + momentum * change
        # take a step
        solution = solution - new_change
        # save the change
        change = new_change
        # evaluate candidate point
        solution_eval = objective(solution)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
        # report progress
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
        #Test to compare convergence
        if solution_eval <= 0.0000000001:
            break 
    print(f"took {i} iterations")
    
    return [solutions, scores]

def test_basicGD():
    # seed the pseudo random number generator
    np.random.seed(4)
    # define range for input
    bounds = np.asarray([[-1.0, 1.0]])
    # define the total iterations
    n_iter = 300
    # define the step size
    step_size = 0.25
    # define momentum
    momentum = 0.10
    # perform the gradient descent search
    solutions, scores = gradient_descentMomentum(objective, derivative, bounds, n_iter, step_size, momentum)
    # sample input range uniformly at 0.1 increments
    inputs = np.arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
    # compute targets
    results = objective(inputs)
    # create a line plot of input vs result
    plt.plot(inputs, results)
    # plot the solutions found
    plt.plot(solutions, scores, '.-', color='red')
    # show the plot
    plt.show()

#test_basicGD()
"""
Convergene takes 16 iterations for learning rate 0.25, but when we 
add momentum of 0.3, it takes 18 iterations. Changing momentum down to
0.1 got the iterations down to 10, which is best possible result we could see.
"""
