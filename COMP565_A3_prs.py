import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Constants
"""

COMPRESSION_GZIP = "gzip"
M = 100
N = 439

TOTAL_EPOCHS = 10

TAU_BETA_INIT = 200.0
TAU_EPS_INIT = 1.0
PI_INIT = 0.01

TAU_BETA_S_INIT = 1.0
MU_BETA_S_INIT = 0.0
GAMMA_S_INIT = 0.01


class Cavi:

    """ Initialize """

    def __init__(self, n, m,
                 tau_beta,
                 tau_eps,
                 pi,
                 tau_beta_s,
                 mu_beta_s,
                 gamma_s,
                 ld,
                 marginal) -> None:

        # Init parameters
        self._tau_beta = tau_beta
        self._tau_eps = tau_eps
        self._pi = pi

        # Init expectations
        self._tau_beta_s = np.full((M), tau_beta_s, dtype=float)  # M x 1
        self._mu_beta_s = np.full((M), mu_beta_s, dtype=float)  # M x 1
        self._gamma_s = np.full((M), gamma_s, dtype=float)  # M x 1

        self._n = n
        self._m = m
        self._ld = ld
        self._marginal = marginal

    """ Train CAVI model """

    def fit(self, total_epochs):

        result = []

        for epoch in range(total_epochs):
            self._calculate_expectation()
            self._update_parameters()
            elbo = self._calculate_elbo()
            result.append(elbo)
            print("Epoch: {}, elbo: {}".format(epoch, elbo))

        return result

    """ E Step """

    def _calculate_expectation(self):

        for j in range(self._m):

            self._tau_beta_s[j] = self._get_tau_beta_j(j)
            self._mu_beta_s[j] = self._get_mu_beta_j(j)
            self._gamma_s[j] = self._get_gamma_j(j)

    """ M Step """

    def _update_parameters(self):

        mu_plus_tau = np.power(self._mu_beta_s, 2) + \
            np.reciprocal(self._tau_beta_s)  # M x 1
        numerator = np.sum(np.multiply(self._gamma_s, mu_plus_tau))  # 1 x 1
        denomenator = np.sum(self._gamma_s)  # 1 x 1
        self._tau_beta = 1 / (numerator / denomenator)
        self._pi = np.sum(self._gamma_s) / self._m

    """ ELBO """

    def _calculate_elbo(self):
        result = None
        return result

    def _elbo_1(self):
        pass

    def _elbo_2(self):
        pass

    def _elbo_3(self):
        pass

    def _elbo_4(self):
        pass

    def _elbo_5(self):
        pass

    """ Interred precision of effect size """

    def _get_tau_beta_j(self, j):
        r_jj = self._ld[j, j]
        return self._n * r_jj * self._tau_eps + self._tau_beta

    """ Interred mean of effect size """

    def _get_mu_beta_j(self, j):

        tau_beta_j_s = self._tau_beta_s[j]
        marginal_j = self._marginal[j]

        sum = 0
        for i in range(self._m):

            if i == j:
                continue

            gamma_i_s = self._gamma_s[i]
            mu_beta_i_s = self._mu_beta_s[i]
            r_ij = self._ld[i, j]
            sum += gamma_i_s * mu_beta_i_s * r_ij

        return self._n * self._tau_eps / tau_beta_j_s * (marginal_j - sum)

    """ Interred PIP of effect size """

    def _get_gamma_j(self, j):

        mu_j = self._get_mu_j(j)
        return 1 / (1 + math.exp(-1 * mu_j))

    def _get_mu_j(self, j):

        tau_beta_j_s = self._tau_beta_s[j]
        mu_beta_j_s = self._mu_beta_s[j]

        return math.log(self._pi / (1 - self._pi)) + 0.5 * math.log(self._tau_beta / tau_beta_j_s) + 0.5 * tau_beta_j_s * math.pow(mu_beta_j_s, 2)


"""
Main
"""


def main():

    # Load data
    ld = pd.read_csv("data/LD.csv.gz", compression=COMPRESSION_GZIP)
    print("LD shape: {} \n{}".format(ld.shape, ld.head()))

    marginal = pd.read_csv("data/beta_marginal.csv.gz",
                           compression=COMPRESSION_GZIP)
    print("Marginal: {} \n{}".format(marginal.shape, marginal.head()))

    assert (ld.iloc[:, 0]).equals(marginal.iloc[:, 0])

    ld = ld.to_numpy()[:, 1:]
    assert ld.shape[0] == M
    assert ld.shape[1] == M

    marginal = marginal.to_numpy()[:, 1]
    assert marginal.shape[0] == M

    cavi = Cavi(N, M, TAU_BETA_INIT, TAU_EPS_INIT, PI_INIT,
                TAU_BETA_S_INIT, MU_BETA_S_INIT, GAMMA_S_INIT, ld, marginal)
    cavi.fit(TOTAL_EPOCHS)


if __name__ == "__main__":
    main()
