from SGD import *
import numpy as np

if __name__ == "__main__":
    n = 100 
    np.random.seed(4)
    x = np.random.rand(n,1)
    y = 4+3*x + x**2 +np.random.randn(n,1)

    x_exact = np.linspace(0,2,11)
    y_exact = 4+3*x_exact + x_exact**2
    #y = 2.0+3*x +4*x*x# +np.random.randn(n,1)
    t0, t1 = 5, 50
    #GD(x,y, 1000, 0.1)
    #SGD_Tuned(x,y, 200, 0.1, 5)
    SGD_Ridge(x,y, 200, 0.1, 5, lmbda=1)
    
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