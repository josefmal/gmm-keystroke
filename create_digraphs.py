import os

RAW_TRAIN_DATA_PATH = "data/s0_baseline_free/"
RAW_VALID_DATA_PATH = "data/s1_baseline_free/"
RAW_TEST_DATA_PATH = "data/s2_baseline_free/"

PROCESSED_DATA_PATH = "processed_data/"
TRAIN_DATA_PATH = "processed_data/train/"
VALID_DATA_PATH = "processed_data/valid/"
TEST_DATA_PATH = "processed_data/test/"

SPECIAL_KEYS = ["LMenu", "Tab", "Space", "Back",
                "LShiftKey", "OemPeriod", "OemQuestion",
                "Oemcomma", "Escape", "LControlKey",
                "RControlKey", "Left", "Right", "Up", "Down",
                "Delete", "Return"]


def parse_raw_data(read_path, write_path, min_delay=10, max_delay=500, session_fraction=1, special_keys=True):
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
        n_lines = len(open(file_name).readlines())

        with open(file_name, "r") as file:

            prev = None  # (key, time)
            new = None  # (key, time)
            for i, line in enumerate(file):

                if i >= n_lines*session_fraction:
                    break

                key, action, time = line.split()

                if action != "KeyDown":
                    continue

                if not special_keys and key in SPECIAL_KEYS:
                    prev = None
                    new = None
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
    if not os.path.exists(VALID_DATA_PATH):
        os.mkdir(VALID_DATA_PATH)

    parse_raw_data(RAW_TRAIN_DATA_PATH, TRAIN_DATA_PATH, min_delay=0, max_delay=1e6, session_fraction=1, special_keys=False)
    parse_raw_data(RAW_VALID_DATA_PATH, VALID_DATA_PATH, min_delay=0, max_delay=1e6, session_fraction=1, special_keys=False)
    parse_raw_data(RAW_TEST_DATA_PATH, TEST_DATA_PATH, min_delay=0, max_delay=1e6, session_fraction=1, special_keys=False)

    print("Data was preprocessed successfully.")


if __name__ == "__main__":
    main()
