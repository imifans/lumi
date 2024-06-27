import classifier
from utils import (
    get_device,
    handle_argv,
    IMUDataset,
    load_classifier_config,
    prepare_classifier_dataset,
)


if __name__ == "__main__":

    training_rate = 0.8  # unlabeled sample / total sample
    label_rate = 0.01  # labeled sample / unlabeled sample
    balance = True

    mode = "base"
    method = "gru"
    args = handle_argv("classifier_" + mode + "_" + method, "train.json", method)

    classifier.main(args, distill=True)
