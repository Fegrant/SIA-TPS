import matplotlib.pyplot as plt
import numpy as np

def biplot(score, names):
    xs = score[:,0] # projection on PC1
    ys = score[:,1] # projection on PC2
    #scalex = 1.0 / (xs.max() - xs.min())
    #scaley = 1.0 / (ys.max() - ys.min())
    scalex = 1.0
    scaley = 1.0
    _, ax = plt.subplots(figsize=(10, 10))

    # plot letters
    ax.scatter(xs * scalex, ys * scaley, color='b', alpha=0.5)
    for i in np.arange(len(names)):
        ax.annotate(names[i], (xs[i] * scalex + 0.015, ys[i] * scaley), color='blue') # letter names
    
    ax.set_xlabel("PC{}".format(1))
    ax.set_ylabel("PC{}".format(2))

    plt.grid()
    plt.show()
