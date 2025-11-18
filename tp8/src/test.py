import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

fsr = 20

v1 = 2
v2 = 14
variance = 1
sigma = np.sqrt(variance)

for i in range(3):
    mu1 = v1 + i*fsr
    mu2 = v2 + i*fsr
    plt.vlines(i*fsr , ymin = 0 , ymax=1 , label = f"{i}" , color = "black")
    x1 = np.linspace(mu1 - 3*sigma, mu1 + 3*sigma, 100)
    x2 = np.linspace(mu2 - 3*sigma, mu2 + 3*sigma, 100)
    plt.plot(x1 ,stats.norm.pdf(x1,mu1,sigma) , color = "blue")
    plt.plot(x2 ,stats.norm.pdf(x1,mu1,sigma) , color = "red")


plt.show()


