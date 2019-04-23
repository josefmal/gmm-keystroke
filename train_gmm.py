import numpy as np
import sklearn.mixture
import os

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
        self.all_users_digraphs_test = [None for user in self.users]

        self.all_users_digraphs_gmms = [None for user in self.users]

    def get_train_data(self, path):

        """
        Reads all the training data as a list of dictionaries (one per user).
        """
        for i, user in enumerate(self.users):
            digraph_dict = self.__get_processed_data(path, user)
            self.all_users_digraphs_train[i] = digraph_dict

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

                self.all_users_digraphs_gmms[i][digraph] = gmm

    def predict(self, S_thresh=None):

        """
        Predicts on the test data (i.e. tries to authenticate every user with every other user
        by presenting the test data).

        Returns the FAR and FRR.

        """
        n_imposters = len(self.users)*(len(self.users) - 1)
        FA_errors = 0
        FR_errors = 0

        if S_thresh is None:
            S_thresh = self.S_thresh

        for query_user_id in self.users:

            for claimed_user_id in self.users:

                S = self.__compute_similarity(query_user_id, claimed_user_id)

                if query_user_id == claimed_user_id:
                    print("Same: {}".format(S))
                # else:
                #    print(S)

                if query_user_id != claimed_user_id:

                    if S >= S_thresh:
                        FA_errors += 1
                else:

                    if S < S_thresh:
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

    def __compute_similarity(self, query_user_id, claimed_user_id):
        """
        Computes similarity between query_user and claimed_user
        using the "Digraph similarity algorithm" (Alg. 1 in paper).

        Returns list of similarities.
        """
        S = 0

        query_ind = self.users.index(query_user_id)
        query_digraph = self.all_users_digraphs_test[query_ind]

        claimed_ind = self.users.index(claimed_user_id)
        claimed_gmms = self.all_users_digraphs_gmms[claimed_ind]

        for key, delays in query_digraph.items():

            if key not in claimed_gmms.keys():
                # TODO : handle this special case
                continue

            gmm = claimed_gmms[key]
            log_likes = gmm.score_samples(np.asarray(delays).reshape(-1, 1))
            S += sum(log_likes)

        return S


def main():

    M = 3  # number of components in GMM
    delta = 1  # similarity metric parameter

    users = os.listdir(TRAIN_DATA_PATH)
    users = list(map(lambda x: x[:3], users))
    # users = ["001", "002", "003", "004", "005"]

    model = GMMKeystrokeModel(users, M, delta, S_thresh=-15000)

    print("Loading training data...")
    model.get_train_data(TRAIN_DATA_PATH)

    print("Fitting GMMs...")
    model.fit()

    print("Loading validation data...")
    model.get_test_data(VALID_DATA_PATH)

    print("Predicting on validation data...")
    FAR, FRR = model.predict()
    print("FAR: {}".format(FAR))
    print("FRR: {}".format(FRR))
    print()

    """
    # Predict on each users test data
    for S_thresh in np.arange(0.1, 1, 0.05):
        FAR, FRR = model.predict(S_thresh)
        print("S_thresh = {}".format(S_thresh))
        print("FAR: {}".format(FAR))
        print("FRR: {}".format(FRR))
        print()
    """


if __name__ == "__main__":
    main()
