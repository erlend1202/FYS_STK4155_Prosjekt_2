from SGD import *
import numpy as np
from matplotlib.colors import LogNorm


def grid_search_hyperparameters(x, y, y_exact, plot_title, func, verbose = False):
    #batch_size = np.linspace(1,10,10)
    if func == GD:
        learning_rates = np.logspace(-5,-1,5)
        momentums = np.logspace(-5,-1,5)
    else:
        learning_rates = np.logspace(-5,1,7)
        momentums = np.logspace(-5,1,7)
    mse_values = np.zeros((len(momentums), len(learning_rates)))

    for i, mom in enumerate(momentums):
        for j, eta in enumerate(learning_rates):
            if func == SGD_Tuned or func == SGD:
                xnew,y_tilde = func(x,y, Niterations=20, momentum=mom, M=5, eta=eta, plot=False)
            else:
                xnew,y_tilde = func(x,y, Niterations=20, momentum=mom, eta=eta, plot=False)
            mse = MSE(y_exact, y_tilde)
            mse_values[i, j] = mse

            if verbose:
                print(f"eta:{eta}, momentum:{mom} gives mse {mse}")

    def array_elements_to_string(arr):
        new_arr = []

        for element in arr:
            new_arr.append(str(element))
        
        return new_arr
    
    def show_values_in_heatmap(heatmap, axes, text_color = "white"):
        for i in range(len(heatmap)):
            for j in range(len(heatmap[0])):
                axes.text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center", color=text_color)

    labels_x = array_elements_to_string(learning_rates)
    labels_y = array_elements_to_string(momentums)

    plt.figure()
    show_values_in_heatmap(mse_values, plt.gca())
    
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(learning_rates)), labels_x)
    plt.yticks(np.arange(0, len(momentums)), labels_y)
    plt.xlabel("$Momentum$")
    plt.ylabel("$\eta$")
    #plt.yscale('log')
    #plt.xscale('log')
    plt.gcf().autofmt_xdate()
    plt.imshow(mse_values, norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"figures/{plot_title}")

def epochs_plot(X, y, y_exact, layers, plot_title, max_epochs, lmda, eta, func, verbose = False):
    mse_values = np.zeros(max_epochs)

    for epoch in range(max_epochs):
        if verbose:
            print(f"Testing epoch {epoch}")
    
    plt.figure()
    plt.title(plot_title)
    plt.plot(mse_values)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(f"figures/{plot_title}")




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
    #grid_search_hyperparameters(x,y,y_exact,"SGD with tuning (momentum and learning rates)", SGD_Tuned, verbose=True)
    #grid_search_hyperparameters(x,y,y_exact,"SGD without tuning (momentum and learning rates)", SGD, verbose=True)
    grid_search_hyperparameters(x,y,y_exact,"GD with tuning (momentum and learning rates)", GD_Tuned, verbose=True)
    grid_search_hyperparameters(x,y,y_exact,"GD without tuning  (momentum and learning rates)", GD, verbose=True)
