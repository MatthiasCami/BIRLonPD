import scipy.stats
import numpy as np


class UniformDist:
    def __init__(self, xmax=1., xmin=None):
        self.xmax = xmax
        self.xmin = - xmax if xmin is None else xmin
        self.prob = 1 / (self.xmax - self.xmin)

    def __call__(self, *args, **kwargs):
        return self.prob

    def __str__(self):
        return 'UniformDist(max={}, min={})'.format(self.xmax, self.xmin)

class IsingDist:
    def __init__(self, coupling = 1, magnetization = 1):
        self.coupling = coupling
        self.magnetization = magnetization
        self.partitions = self.get_partitions()

    def get_partitions(self):
        bases = [0,1,4,5]
        additional = [0,2,8,10]
        partitions = {}
        for i in bases:
            new_partition = []
            for k in additional:
                new_partition.append(i + k)
            partitions[i] = new_partition
        return partitions

    def calculate_sum_over_couples(self, R):
        sum = 0
        for key in self.partitions.keys():
            partition = self.partitions[key]
            for x in partition:
                for y in partition:
                    if (x != y):
                        sum += R[x]*R[y]
        return sum

    def __call__(self, R):
        return np.exp(-self.coupling*self.calculate_sum_over_couples(R) - self.magnetization*np.sum(R))

class OpposingDist:
    def __init__(self, coupling = 3, magnetization = 0):
        self.coupling = coupling
        self.magnetization = magnetization
        self.partitions = self.get_partitions()
    def get_partitions(self):
        bases = [0,1,4,5]
        additional = [0,2,8,10]
        partitions = {}
        for i in bases:
            new_partition = []
            for k in additional:
                new_partition.append(i + k)
            partitions[i] = new_partition
        return partitions

    def calculate_sum_over_couples(self, R):
        sum = 0
        for x in range(0,15,2):
            sum += np.abs(R[x] - R[x+1])
        return sum

    def calculate_sum_over_couples2(self, R):
        sum = 0
        for key in self.partitions.keys():
            partition = self.partitions[key]
            for x in partition:
                for y in partition:
                    if (x != y):
                        sum += R[x]*R[y]
        return sum

    def __call__(self, R):
        return np.exp(-self.coupling*self.calculate_sum_over_couples(R) -self.coupling*self.calculate_sum_over_couples2(R) - self.magnetization*np.sum(R))

class DistBase:
    def __init__(self, dist, params):
        self.dist = dist
        self.params = params

    def __call__(self, x):
        """
        :x: input
        :return: P(x)
        """
        return np.exp(np.sum(self.dist.logpdf(x, **self.params)))

    def sample(self, size=10):
        return self.dist.rvs(size=size, **self.params)

    def __str__(self):
        return self.__class__.__name__ + '(' + ', '.join(['{}={}'.format(key, value)
                                                        for key, value in self.params.items()]) + ')'


class GaussianDist(DistBase):
    def __init__(self, loc=0, scale=0.1):
        """
        :param loc: location of gaussian distribution
        :param scale: var == scale ** 2
        """
        params = dict(loc=loc, scale=scale)
        dist = scipy.stats.norm
        super().__init__(dist=dist, params=params)



class BetaDist(DistBase):
    def __init__(self, a=0.5, b=0.5, loc=0, scale=1):
        params = dict(a=a, b=b, loc=loc, scale=scale)
        dist = scipy.stats.beta
        super().__init__(dist=dist, params=params)


class GammaDist(DistBase):
    def __init__(self, a=2, loc=0, scale=1):
        params = dict(a=a, loc=loc, scale=scale)
        dist = scipy.stats.gamma
        super().__init__(dist=dist, params=params)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    dists = (GaussianDist, BetaDist, GammaDist)
    for dist in dists:
        distribution = dist()
        samples = distribution.sample(size=100)
        plt.hist(samples)
        plt.title(distribution)
        #path = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-3], 'results',
        #                          '{}.png'.format(dist.__name__))
        path_temp = "C:\\Users\\Matthias\\Documents\\Thesis\\Bayesian\\bayesian_irl-master\\results"
        path=os.path.join(path_temp, '{}.png'.format(dist.__name__) )
        plt.savefig(path)
        plt.cla()
