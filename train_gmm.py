import os

import numpy as np
import matplotlib.pyplot as plt
import sklearn.mixture
from tqdm import tqdm

PROCESSED_DATA_PATH = "processed_data/"
TRAIN_DATA_PATH = "processed_data/train/"
VALID_DATA_PATH = "processed_data/valid/"
TEST_DATA_PATH = "processed_data/test/"


class GMMKeystrokeModel(object):

    def __init__(self, users, M, delta, S_thresh):

        """
        Initializes the model.

        users = List of user ids in the system
        M = Number of components in each GMM
        delta = Similarity tolerance parameter
        S_thresh = Threshold for evaluating

        """
        self.users = users
        self.M = M
        self.delta = delta
        self.S_thresh = S_thresh

        self.all_users_digraphs_train = [None for user in self.users]
        self.all_users_digraphs_valid = [None for user in self.users]
        self.all_users_digraphs_test = [None for user in self.users]

        self.all_users_digraphs_gmms = [None for user in self.users]

    def get_train_data(self, path):

        """
        Reads all the training data as a list of dictionaries (one per user).
        """
        for i, user in enumerate(self.users):
            digraph_dict = self.__get_processed_data(path, user)
            self.all_users_digraphs_train[i] = digraph_dict

    def get_valid_data(self, path):

        """
        Reads all the validation data as a list of dictionaries (one per user).
        """
        for i, user in enumerate(self.users):
            digraph_dict = self.__get_processed_data(path, user, keep_small=False)
            self.all_users_digraphs_valid[i] = digraph_dict

    def get_test_data(self, path):

        """
        Reads all the test data as a list of dictionaries (one per user).
        """
        for i, user in enumerate(self.users):
            digraph_dict = self.__get_processed_data(path, user, keep_small=False)
            self.all_users_digraphs_test[i] = digraph_dict

    def fit(self):

        """
        Fits GMMs to the training data.
        """

        if self.all_users_digraphs_train[0] is None:
            raise ValueError("No training data loaded. Load training data with get_train_data(self, path).")

        for i, user in enumerate(self.users):

            user_digraphs = self.all_users_digraphs_train[i]

            self.all_users_digraphs_gmms[i] = {}
            for digraph in user_digraphs.keys():

                delays = user_digraphs[digraph]
                gmm = sklearn.mixture.GaussianMixture(n_components=self.M)
                gmm.fit(delays.reshape(-1, 1))

                self.all_users_digraphs_gmms[i][digraph] = (gmm.means_, gmm.covariances_, gmm.weights_)

    def calculate_scores(self):

        """
        Calculates score when every user queries every other user (and itself).

        """
        imposter_scores = []
        valid_scores = []

        for query_user_id in self.users:

            for claimed_user_id in self.users:

                S = self.__compute_similarity(query_user_id, claimed_user_id, "valid")

                if query_user_id != claimed_user_id:
                    imposter_scores.append(S)
                else:
                    valid_scores.append(S)

        return valid_scores, imposter_scores

    def predict(self):

        """
        Predicts on the test data (i.e. tries to authenticate every user with every other user
        by presenting the test data).

        Returns the FAR and FRR.

        """
        n_imposters = len(self.users)*(len(self.users) - 1)
        FA_errors = 0
        FR_errors = 0

        for query_user_id in self.users:

            for claimed_user_id in self.users:

                S = self.__compute_similarity(query_user_id, claimed_user_id, "test")

                if query_user_id != claimed_user_id:

                    if S >= self.S_thresh:
                        FA_errors += 1
                else:

                    if S < self.S_thresh:
                        FR_errors += 1

        FAR = float(FA_errors)/n_imposters
        FRR = float(FR_errors)/len(self.users)

        return FAR, FRR

    def __get_processed_data(self, path, user_id, keep_small=False):

        """
        Reads preprocessed data of user with user_id from path.
        Returns the digraph as a dictionary, in which each entry has the form:

        "wo" : (124, 75, 242, ...)

        If keep_small is False, any digraph with less than M samples is discarded.

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

        if not keep_small:

            for key, entry in list(digraph_dict.items()):
                if entry.shape == ():
                    del digraph_dict[key]
                elif entry.shape[0] < 10:
                    del digraph_dict[key]
                elif len(set(entry)) < self.M:
                    del digraph_dict[key]

        return digraph_dict

    def __compute_similarity(self, query_user_id, claimed_user_id, data_type):
        """
        Computes similarity between query_user and claimed_user
        using the "Digraph similarity algorithm" (Alg. 1 in paper).

        Returns list of similarities.
        """
        count = 0
        total_count = 0

        query_ind = self.users.index(query_user_id)
        if data_type == "valid":
            query_digraph = self.all_users_digraphs_valid[query_ind]
        elif data_type == "test":
            query_digraph = self.all_users_digraphs_test[query_ind]

        claimed_ind = self.users.index(claimed_user_id)
        claimed_gmm_params = self.all_users_digraphs_gmms[claimed_ind]

        for key, delays in query_digraph.items():

            if key not in claimed_gmm_params.keys():
                # TODO : handle this special case
                continue

            means, covars, weights = claimed_gmm_params[key]

            for delay in delays:

                total_count += 1

                for i in range(self.M):

                    mean = np.asscalar(means[i])
                    covar = np.asscalar(covars[i])
                    weight = np.asscalar(weights[i])

                    if delay >= mean - self.delta*np.sqrt(covar) and delay <= mean + self.delta*np.sqrt(covar):
                        count += weight

        S = count/float(total_count)

        return S


def compute_FAR(imposter_scores, S_thresh):

    FA_errors = 0
    for score in imposter_scores:
        if score >= S_thresh:
            FA_errors += 1

    return float(FA_errors)/len(imposter_scores)


def compute_FRR(valid_scores, S_thresh):
    FR_errors = 0
    for score in valid_scores:
        if score < S_thresh:
            FR_errors += 1

    return float(FR_errors)/len(valid_scores)


def main():

    M = 3  # number of components in GMM
    delta = 1  # similarity metric parameter

    users = os.listdir(TRAIN_DATA_PATH)
    users = list(map(lambda x: x[:3], users))
    # users = ["001", "002", "003", "004", "005"]

    model = GMMKeystrokeModel(users, M, delta, S_thresh=0.32)

    print("Loading training data...")
    model.get_train_data(TRAIN_DATA_PATH)

    print("Fitting GMMs...")
    model.fit()

    print("Loading validation data...")
    model.get_valid_data(VALID_DATA_PATH)

    print("Searching for best S threshold...")
    valid_scores, imposter_scores = model.calculate_scores()
    best_S_thresh = 0
    best_sum = np.float("inf")
    all_S_thresh = []
    all_FAR = []
    all_FRR = []
    for S_thresh in tqdm(np.arange(0, 1.05, 0.01)):
        FAR = compute_FAR(imposter_scores, S_thresh)
        FRR = compute_FRR(valid_scores, S_thresh)
        error_sum = FAR + FRR
        if error_sum < best_sum:
            best_S_thresh = S_thresh
            best_sum = error_sum

        all_S_thresh.append(S_thresh)
        all_FAR.append(FAR)
        all_FRR.append(FRR)

    plt.figure()
    plt.plot(all_FAR, all_FRR)
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("FRR/FAR for different thresholds")
    plt.savefig("GMM_FRR_FRR_plot.png")
    plt.show()

    # Use the best S_thresh
    model.S_thresh = best_S_thresh

    print("Loading test data...")
    model.get_test_data(TEST_DATA_PATH)

    # Predict on each users test data
    print("Predicting on test data...")
    FAR, FRR = model.predict()
    print("S_thresh = {}".format(model.S_thresh))
    print("FAR: {}".format(FAR))
    print("FRR: {}".format(FRR))
    print()


if __name__ == "__main__":
    main()
