import numpy as np
import sklearn.mixture

PROCESSED_DATA_PATH = "processed_data/"
TRAIN_DATA_PATH = "processed_data/train/"
TEST_DATA_PATH = "processed_data/test/"


def read_processed_data(path, user_id, M=0):
    """
    Reads the processed data (train or test) for one user and returns
    the digraph as a dictionary, where each entry has the form:

    "wo" : (124, 75, 242, ...)

    """

    file_name = path + user_id + ".txt"
    digraph_dict = {}
    with open(file_name, "r") as file:
        for line in file:
            keys, delay = line.split()
            if keys in digraph_dict.keys():
                digraph_dict[keys] = np.append(digraph_dict[keys], float(delay))
            else:
                digraph_dict[keys] = np.array(float(delay))

    # Delete any entries with less than M samples
    for key, entry in list(digraph_dict.items()):
        if entry.shape == ():
            del digraph_dict[key]
        elif entry.shape[0] < M:
            del digraph_dict[key]
        elif len(set(entry)) < M:
            del digraph_dict[key]

    return digraph_dict


def fit_gmm(digraph_delays, M):
    """
    Fits a GMM to the set of delays for one digraph.
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=M)
    gmm.fit(digraph_delays.reshape(-1, 1))

    return gmm


def compute_similarites(user_digraphs, user_gmm_params, M, delta):
    """
    Computes similarity metric using the "Digraph similarity algorithm"
    """
    s = []

    for i in range(M):

        count = 0

        for keys, delays in user_digraphs.items():

            for delay in delays:

                if keys not in user_gmm_params.keys():
                    continue

                means, covars, weights = user_gmm_params[keys]
                mean = np.asscalar(means[i])
                covar = np.asscalar(covars[i])

                if delay > mean - delta*np.sqrt(covar) and delay < mean + delta*np.sqrt(covar):
                    count += 1

        s.append(count*weights[i])

    return s


def main():

    M = 2  # number of components in GMM
    delta = 0.9  # similarity metric parameter

    users = ["001", "002", "003", "004", "005"]
    user_gmm_params = []

    # Fit a GMM to each user's training data
    for user in users:

        user_digraphs = read_processed_data(TRAIN_DATA_PATH, user, M)

        digraph_gmms = {}
        for digraph in user_digraphs.keys():

            fitted_gmm = fit_gmm(user_digraphs[digraph], M)
            digraph_gmms[digraph] = (fitted_gmm.means_, fitted_gmm.covariances_, fitted_gmm.weights_)

        user_gmm_params.append(digraph_gmms)

    # Predict on each users test data
    for i, gmm_params in enumerate(user_gmm_params):
        for user in users:

            user_digraphs = read_processed_data(TEST_DATA_PATH, user)

            s = compute_similarites(user_digraphs, gmm_params, M, delta)

            print("Similarity of user {} and user {}: {}\t Norm: {}".format(users[i], user, s, np.linalg.norm(s)))

        print()


if __name__ == "__main__":
    main()
