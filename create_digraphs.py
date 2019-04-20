import os

RAW_TRAIN_DATA_PATH = "data/baseline_s0/"
RAW_TEST_DATA_PATH = "data/baseline_s1/"

PROCESSED_DATA_PATH = "processed_data/"
TRAIN_DATA_PATH = "processed_data/train/"
TEST_DATA_PATH = "processed_data/test/"


def parse_raw_data(read_path, write_path, min_delay=50, max_delay=200):
    """
    Parses the raw data at read_path into sequences of digraphs on the form:
    [("wo", 323), ("or", 200), ("rd", 432)]
    where the second tuple entry is the KeyDown to KeyDown delay.

    Any digraph with delay < min_delay, or delay > max_delay is rejected.

    Writes resulting sequence to write_path.
    """
    # Extract digraphs from raw file
    file_names = os.listdir(read_path)
    for file_name in file_names:

        user_id = file_name[:3]
        users_digraphs = []

        file_name = read_path + file_name
        with open(file_name, "r") as file:

            prev = None  # (key, time)
            new = None  # (key, time)
            for line in file:

                key, action, time = line.split()

                if action != "KeyDown":
                    continue

                if prev is None:
                    prev = (key, time)
                    new = (key, time)
                    continue

                new = (key, time)
                keys = prev[0] + new[0]
                delay = int(new[1]) - int(prev[1])

                prev = new

                if not(delay < min_delay or delay > max_delay):
                    users_digraphs.append((keys, delay))

        # Write processed data to file
        write_file = write_path + user_id + ".txt"
        with open(write_file, "w+") as file:
            for entry in users_digraphs:
                file.write(entry[0] + " " + str(entry[1]) + "\n")


def main():

    if os.path.exists(PROCESSED_DATA_PATH):
        ans = input("All preprocessed data will be overwritten. Do you want to continue? (Y/n) >> ")
        if not(ans == "" or ans.lower() == "y" or ans.lower() == "yes"):
            exit()

    # Creates paths for the preprocessed data
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.mkdir(PROCESSED_DATA_PATH)
    if not os.path.exists(TRAIN_DATA_PATH):
        os.mkdir(TRAIN_DATA_PATH)
    if not os.path.exists(TEST_DATA_PATH):
        os.mkdir(TEST_DATA_PATH)

    parse_raw_data(RAW_TRAIN_DATA_PATH, TRAIN_DATA_PATH, min_delay=0, max_delay=1e6)
    parse_raw_data(RAW_TEST_DATA_PATH, TEST_DATA_PATH, min_delay=0, max_delay=1e6)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()
