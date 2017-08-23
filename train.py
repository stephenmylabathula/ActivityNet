import data_provider
import matplotlib.pyplot as plt
import numpy as np

X, Y = data_provider.GenerateInputOutputData()

for i in range(100):
    n = np.random.randint(1000)
    plt.plot(X[n][0])
    plt.title(Y[n])
    plt.show()
    plt.close('all')

print "Done."
