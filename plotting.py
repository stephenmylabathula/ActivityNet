import io
import itertools
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


# Plot Confusion Matrix
def PlotConfusionMatrix(confusion_matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.YlGn):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(confusion_matrix)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf
