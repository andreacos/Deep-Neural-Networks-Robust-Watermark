import matplotlib.pyplot as plt
import numpy as np


class Distribution:

    def __init__(self, name, unwat_model=None, mu=None, sigma=None, sigma_multiplier=2, start=None, end=None, size=None, key=None, output_png=None):
        self._name = name
        self._unwat_model = unwat_model
        self._key = key
        self._mu = mu
        self._sigma = sigma
        self._sigma_mult = sigma_multiplier
        self._start = start
        self._end = end
        self._size = size
        self._png = output_png
        self._values = []
        return

    def GetName(self):
        """
        Getter for the distribution name
        :return: distribution name
        """
        return self._name

    def UpdateParameters(self, name=None, mu=None, sigma=None, start=None, end=None, size=None, key=None, output_png=None):
        """
        Update class member values (useful when using the same object to generate samples from different distributions
        (or the same distribution with different parameters) while storing them in self._values
        :param name: name of the distribution from which to draw
        :param mu: mean of the distribution
        :param sigma: variance of the distribution
        :param size: number of samples drawn from the distribution
        :param key: random seed initializer for reproducibility
        :param output_png: path to the PNG image with the histogram of the samples
        :return: Nothing
        """
        self._name = name if name is not None else self._name
        self._key = key if key is not None else self._key
        self._mu = mu if mu is not None else self._mu
        self._sigma = sigma if sigma is not None else self._sigma
        self._size = size if size is not None else self._size
        self._png = output_png if output_png is not None else self._png
        return

    def GetDistribution(self):
        """
        Draws random values from a distribution by its name
        :return: Random samples from the chosen distribution
        """
        if self._name not in ['laplace']:
            raise AssertionError(f"Unsupported distribution {self._name}")

        if self._key is not None:
            np.random.seed(self._key)
        values = self.Histogram(np.random.laplace(loc=self._mu, scale=self._sigma, size=self._size))
        print(f"Variance for watermark values is {np.var(values)}")
        return values

    def GenerateValues(self, name=None, mu=None, sigma=None, size=None, key=None, output_png=None):
        """
        Stores a new array of values drawn from the chosen distribution into the member class array of distributions.
        Allows to update class members before generating the samples in order to change the distribution from which they
        are drawn and/or its parameters. Leave set to None a parameter if the chosen distribution does not require it
        :param name: name of the distribution from which to draw
        :param mu: mean of the distribution
        :param sigma: variance of the distribution
        :param size: number of samples drawn from the distribution
        :param key: random seed initializer for reproducibility
        :param output_png: path to the PNG image with the histogram of the samples
        :return: Nothing
        """
        self.UpdateParameters(size=size, name=name, mu=mu, sigma=sigma, key=key, output_png=output_png)

        if self._name not in ['laplace']:
            raise AssertionError(f"Unsupported distribution {self._name}")

        self._values.append(self.GetDistribution())

        return

    def ResetValues(self):
        """
        Reset the values of the array of random draws
        :return: Nothing
        """
        self._values = []
        return

    def Histogram(self, x):
        """
        Plots the histogram of random samples from a chosen distribution
        :param x: array of sample values
        :return:
        """
        if self._png is not None and isinstance(self._png, str):
            plt.hist(x, density=True, histtype='stepfilled', alpha=0.2, bins=100)
            plt.savefig(self._png)
            plt.close()
        return x
