import numpy as np
import matplotlib.pyplot as plt


class GP:
    def __init__(self, num):
        self.num = num
        self.samples = np.arange(0, 10.0, 10.0 / self.num).reshape(-1, 1)
        self.mu = np.zeros_like(self.samples)
        self.cov = self.kernel(self.samples, self.samples)

    def visualized(self, num_gp_samples=5):
        gp_samples = np.random.multivariate_normal(mean=self.mu.ravel(),cov=self.cov,size=num_gp_samples)
        sample = self.samples.ravel()
        plt.figure()
        for i, gp_sample in enumerate(gp_samples):
            plt.plot(sample, gp_sample, lw=1, ls='--', label=f'Sample {i + 1}')
        plt.legend()
        plt.show()

    def kernel(self,x1, x2, l=0.5):
        dist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return np.exp(-0.5 / l ** 2 * dist)


gp = GP(num=100)
gp.visualized()


