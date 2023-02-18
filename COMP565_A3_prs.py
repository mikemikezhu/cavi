import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr

"""
Constants
"""

COMPRESSION_GZIP = "gzip"
M = 100
N = 439

CAUSAL_SNPS = ["rs9482449", "rs7771989", "rs2169092"]

TOTAL_EPOCHS = 10

TAU_BETA_INIT = 200.0
TAU_EPS_INIT = 1.0
PI_INIT = 0.01

TAU_BETA_S_INIT = 1.0
MU_BETA_S_INIT = 0.0
GAMMA_S_INIT = 0.01

"""
CAVI model
"""


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
            self._calculate_expectation()  # E Step
            self._update_parameters()  # M Step
            elbo = self._calculate_elbo()  # ELBO
            result.append(elbo)
            print("Epoch: {}, elbo: {}".format(epoch, elbo))

        return result

    """ Q4: PRS """

    def predict(self, x):
        return np.matmul(x, np.multiply(self._gamma_s, self._mu_beta_s))

    """ Q5: Inferred PIP """

    def get_inferred_pip(self):
        return self._gamma_s

    """ Q1: E Step """

    def _calculate_expectation(self):

        for j in range(self._m):

            self._tau_beta_s[j] = self._get_tau_beta_j(j)
            self._mu_beta_s[j] = self._get_mu_beta_j(j)
            self._gamma_s[j] = self._get_gamma_j(j)

    def _get_tau_beta_j(self, j):
        # Inferred precision of effect size
        r_jj = self._ld[j, j]
        return (self._n * r_jj) * self._tau_eps + self._tau_beta

    def _get_mu_beta_j(self, j):

        # Inferred mean of effect size
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

    def _get_gamma_j(self, j):

        # Inferred PIP of effect size
        mu_j = self._get_mu_j(j)
        result = 1 / (1 + math.exp(-1 * mu_j))
        result = 0.01 if result < 0.01 else result
        result = 0.99 if result > 0.99 else result
        return result

    def _get_mu_j(self, j):

        tau_beta_j_s = self._tau_beta_s[j]
        mu_beta_j_s = self._mu_beta_s[j]

        return math.log(self._pi / (1 - self._pi)) + 0.5 * math.log(self._tau_beta / tau_beta_j_s) + 0.5 * tau_beta_j_s * math.pow(mu_beta_j_s, 2)

    """ Q2: M Step """

    def _update_parameters(self):

        mu_plus_tau = np.power(self._mu_beta_s, 2) + \
            np.reciprocal(self._tau_beta_s)  # M x 1
        numerator = np.sum(np.multiply(self._gamma_s, mu_plus_tau))  # 1 x 1
        denomenator = np.sum(self._gamma_s)  # 1 x 1
        self._tau_beta = 1 / (numerator / denomenator)
        self._pi = np.sum(self._gamma_s) / self._m

    """ Q3: ELBO """

    def _calculate_elbo(self):
        return self._elbo_1() + self._elbo_2() + self._elbo_4() - self._elbo_3() - self._elbo_5()

    def _elbo_1(self):

        part_1 = 0.5 * self._n * math.log(self._tau_eps)
        part_2 = 0.5 * self._tau_eps * self._n
        part_3 = self._tau_eps * \
            (np.dot(np.multiply(self._gamma_s, self._mu_beta_s), self._n * self._marginal))

        part_4 = 0
        for j in range(self._m):
            gamma_j_s = self._gamma_s[j]
            mu_beta_j_s = self._mu_beta_s[j]
            tau_beta_j_s = self._tau_beta_s[j]
            r_jj = self._ld[j, j]
            part_4 += 0.5 * self._tau_eps * \
                (gamma_j_s * (mu_beta_j_s ** 2 + (1 / tau_beta_j_s))) * (self._n * r_jj)

        part_5 = 0
        for j in range(self._m):
            gamma_j_s = self._gamma_s[j]
            mu_beta_j_s = self._mu_beta_s[j]
            for k in range(j + 1, self._m):
                gamma_k_s = self._gamma_s[k]
                mu_beta_k_s = self._mu_beta_s[k]
                r_kj = self._ld[k, j]
                part_5 += gamma_j_s * mu_beta_j_s * \
                    gamma_k_s * mu_beta_k_s * (self._n * r_kj)
        part_5 *= self._tau_eps

        return part_1 - part_2 + part_3 - part_4 - part_5

    def _elbo_2(self):

        part_1 = self._m * (-0.5 * math.log(2 * self._pi)
                            * (1 / self._tau_beta))
        part_2 = 0
        for j in range(self._m):
            gamma_j_s = self._gamma_s[j]
            mu_beta_j_s = self._mu_beta_s[j]
            tau_beta_j_s = self._tau_beta_s[j]
            part_2 += 0.5 * self._tau_beta * gamma_j_s * \
                (mu_beta_j_s ** 2 + (1 / tau_beta_j_s))

        return part_1 - part_2

    def _elbo_3(self):

        part_1 = self._m * (-0.5 * math.log(2 * self._pi)
                            * (1 / self._tau_beta))
        part_2 = 0.5 * np.sum(self._gamma_s) * math.log(self._tau_beta)

        return part_1 - part_2

    def _elbo_4(self):

        result = 0
        for j in range(self._m):
            gamma_j_s = self._gamma_s[j]
            result += gamma_j_s * \
                math.log(self._pi) + (1 - gamma_j_s) * math.log(1 - self._pi)
        return result

    def _elbo_5(self):

        result = 0
        for j in range(self._m):
            gamma_j_s = self._gamma_s[j]
            result += gamma_j_s * \
                math.log(gamma_j_s) + (1 - gamma_j_s) * math.log(1 - gamma_j_s)
        return result


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

    total_snps = marginal.iloc[:, 0].to_numpy()
    assert len(total_snps) == M

    x_train = pd.read_csv("data/X_train.csv.gz", compression=COMPRESSION_GZIP)
    print("x_train shape: {} \n{}".format(x_train.shape, x_train.head()))

    y_train = pd.read_csv("data/y_train.csv.gz", compression=COMPRESSION_GZIP)
    print("y_train shape: {} \n{}".format(y_train.shape, y_train.head()))

    x_test = pd.read_csv("data/X_test.csv.gz", compression=COMPRESSION_GZIP)
    print("x_test shape: {} \n{}".format(x_test.shape, x_test.head()))

    y_test = pd.read_csv("data/y_test.csv.gz", compression=COMPRESSION_GZIP)
    print("y_test shape: {} \n{}".format(y_test.shape, y_test.head()))

    x_train = x_train.to_numpy()[:, 1:].astype(float)
    y_train = y_train.to_numpy()[:, 1].astype(float)
    x_test = x_test.to_numpy()[:, 1:].astype(float)
    y_test = y_test.to_numpy()[:, 1].astype(float)

    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("x_test shape: {}".format(x_test.shape))
    print("y_test shape: {}".format(y_test.shape))

    ld = ld.to_numpy()[:, 1:]
    assert ld.shape[0] == M
    assert ld.shape[1] == M

    marginal = marginal.to_numpy()[:, 1]
    assert marginal.shape[0] == M

    # Train CAVI model
    cavi = Cavi(N, M, TAU_BETA_INIT, TAU_EPS_INIT, PI_INIT,
                TAU_BETA_S_INIT, MU_BETA_S_INIT, GAMMA_S_INIT, ld, marginal)
    elbo = cavi.fit(TOTAL_EPOCHS)

    # Plot ELBO
    plt.clf()
    plt.scatter(range(TOTAL_EPOCHS), elbo)
    plt.ylabel("ELBO")
    plt.xlabel("Iteration")
    plt.title("Evidence lower bound as a function of EM iteration")
    plt.grid(0.5)
    plt.savefig("elbo.png")
    plt.close()

    # Predict PRS
    y_train_hat = cavi.predict(x_train)
    y_test_hat = cavi.predict(x_test)

    # Calculate PCC
    pcc_train = pearsonr(y_train_hat, y_train).statistic
    pcc_test = pearsonr(y_test_hat, y_test).statistic

    print("Peason correlation coefficient between y_train_hat and y_train: {:.2f}".format(
        pcc_train))
    print("Peason correlation coefficient between y_test_hat and y_test: {:.2f}".format(
        pcc_test))

    # Plot PRS
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    fig.suptitle("PRS prediction on training and testing set")
    ax1.title.set_text(
        'Train (Pearson Correlation Coef. = {:.2f})'.format(pcc_train))
    ax1.scatter(y_train_hat, y_train, s=8.0)
    p_1 = np.poly1d(np.polyfit(y_train_hat, y_train, 1))
    ax1.plot(y_train_hat, p_1(y_train_hat), "r")
    ax1.set_ylabel("True phenotype")
    ax1.set_xlabel("Predict phenotype")
    ax1.grid(0.5)
    ax2.title.set_text(
        'Test (Pearson Correlation Coef. = {:.2f})'.format(pcc_test))
    ax2.scatter(y_test_hat, y_test, s=8.0)
    p_2 = np.poly1d(np.polyfit(y_test_hat, y_test, 1))
    ax2.plot(y_test_hat, p_2(y_test_hat), "r")
    ax2.set_xlabel("Predict phenotype")
    ax2.grid(0.5)
    fig.savefig("prs.png")
    plt.close()

    # Evaluate fine-mapping
    pips = cavi.get_inferred_pip()

    causal_index = np.empty((len(CAUSAL_SNPS)), dtype=int)
    for i, snp in enumerate(CAUSAL_SNPS):
        index = np.argwhere((total_snps == snp))[0, 0]
        causal_index[i] = index

    total_snps_index = np.arange(M)
    non_causal_index = np.delete(total_snps_index, obj=causal_index)

    causal_pips = pips[causal_index]
    non_causal_pips = pips[non_causal_index]

    plt.clf()
    plt.scatter(causal_index, causal_pips,
                c="red", label="True causal SNP")
    plt.scatter(non_causal_index, non_causal_pips,
                c="#2b70ad", label="Non-causal SNP", alpha=0.3)
    plt.ylabel("PIP")
    plt.xlabel("SNP")
    plt.title("Inferred PIP. Causal SNPs are colorred in red")
    plt.grid(0.5)
    plt.legend()
    plt.savefig("pip.png")
    plt.close()


if __name__ == "__main__":
    main()
