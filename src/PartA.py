from SGD import *
from matplotlib.colors import LogNorm
from grid_search import *

if __name__ == "__main__":
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)

    x_exact = np.linspace(0,1,n)
    y_exact = 4+3*x_exact + x_exact**2
    #y = 2.0+3*x +4*x*x# +np.random.randn(n,1)
    t0, t1 = 5, 50
    #GD(x,y, 1000, 0.1)
    #test_tuning(x,y,x_exact,y_exact)
    #xnew, y_pred = SGD_Tuned(x,y, 200, 0.1, 5, plot=False)
    #print(MSE(y_pred, y_exact))
    #plt.plot(x_exact, y_exact)
    #plt.plot(xnew,y_pred)
    #plt.show()
    
    #SGD_Ridge(x,y, 200, 0.1, 5, lmbda=1)
    
    #Task A.3
    def testSGD():

        iterations = [10,50,100,200,1000]
        batch_size = [2,5,10,20,50]
        momentums = [0.05,0.1, 0.2, 0.4, 0.6]
        
        fig = plt.figure()
        plt.plot(x,y,'ro')
        plt.plot(x_exact,y_exact, 'k--', label="y_exact", zorder=100)
        for iter in iterations:
            xnew, ypred = SGD(x,y, iter, 0.1, 5, False)
            plt.plot(xnew, ypred, label=f"num_iterations {iter}")
            plt.legend()
        plt.savefig("figures/taskA3_iter.png")


        fig = plt.figure()
        plt.plot(x,y,'ro')
        plt.plot(x_exact,y_exact, 'k--', label="y_exact", zorder=100)
        for M in batch_size:
            xnew, ypred = SGD(x,y, 200, 0.1, M, False)
            plt.plot(xnew, ypred, label=f"batch size {M}")
            plt.legend()
        plt.savefig("figures/taskA3_batchsize.png")
        
        fig = plt.figure()
        plt.plot(x,y,'ro')
        plt.plot(x_exact,y_exact, 'k--', label="y_exact", zorder=100)
        for momentum in momentums:
            xnew, ypred = SGD(x,y, 200, momentum, 5, False)
            plt.plot(xnew, ypred, label=f"momentum {momentum}")
            plt.legend()
        plt.savefig("figures/taskA3_momentum.png")

    #Task A.5 - testing momentum and learning rates
    grid_search_hyperparameters_SGD(x,y,y_exact,"SGD with tuning (momentum and learning rates)", SGD_Tuned, verbose=True)
    grid_search_hyperparameters_SGD(x,y,y_exact,"SGD without tuning (momentum and learning rates)", SGD, verbose=True)
    grid_search_hyperparameters_SGD(x,y,y_exact,"GD with tuning (momentum and learning rates)", GD_Tuned, verbose=True)
    grid_search_hyperparameters_SGD(x,y,y_exact,"GD without tuning  (momentum and learning rates)", GD, verbose=True)
    grid_search_hyperparameters_SGD_epochs(x,y,y_exact,"SGD without tuning (epochs and batchsize)", SGD, verbose=True)
    grid_search_hyperparameters_SGD_epochs(x,y,y_exact,"SGD with tuning (epochs and batchsize)", SGD_Tuned, verbose=True)
    grid_search_hyperparameters_ridge(x,y,y_exact,"SGD with Ridge (epochs and lambda)", SGD_Ridge, verbose=True)