from root.final.detector.detector_utils import final_detector_cnn, final_detector_mlp


def create_detector_train_test(model_name, csv_number):

    final_detector_cnn(model_name, csv_number)
