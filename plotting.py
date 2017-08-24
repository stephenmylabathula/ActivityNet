import numpy as np
import matplotlib.pyplot as plt


# Plot Action Label Data (after windowing) Atop Signal Data
def PlotSequentialInputOutputWindowedData(X, Y, window_size, stride):
    plt.plot(np.hstack([X[i][0, :, 0] for i in range(0, len(X), stride)]))
    plt.plot(np.repeat([Y[i]*10 for i in range(0, len(Y), stride)], window_size))
    plt.show()


# Randomly Plot a Select Number of Action Windows
def PlotSignalWindowLabels(X, Y, num_plots=10):
    for i in range(num_plots):
        n = np.random.randint(X.shape[0])
        plt.plot(X[n][0, :, 0])
        plt.title(np.argmax(Y[n]))
        plt.show()
        plt.close('all')
