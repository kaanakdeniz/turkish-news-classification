from data.make_dataset import make_dataset
from preprocess.build_data import build_data
from preprocess.transform_data import transform_data
from models.train_model import train_model
from models.test_model import test_model
from models.predict_model import predict
from visualization.visualize import (
    create_freq_chart,
    create_wordclouds,
    create_results_chart,
)


if __name__ == "__main__":
    # make_dataset()
    # build_data()
    # create_freq_chart()
    # create_wordclouds()
    # transform_data()

    # train_model(algorithm="svm", method="bag_of_words")
    # test_model(algorithm="svm", method="bag_of_words")
    # train_model(algorithm="knn", method="bag_of_words")
    # test_model(algorithm="knn", method="bag_of_words")
    # train_model(algorithm="naive_bayes", method="bag_of_words")
    # test_model(algorithm="naive_bayes", method="bag_of_words")
    # train_model(algorithm="linear_regression", method="bag_of_words")
    # test_model(algorithm="linear_regression", method="bag_of_words")
    # train_model(algorithm="random_forest", method="bag_of_words")
    # test_model(algorithm="random_forest", method="bag_of_words")
    # train_model(algorithm="svm", method="tf_idf")
    # test_model(algorithm="svm", method="tf_idf")
    # train_model(algorithm="knn", method="tf_idf")
    # test_model(algorithm="knn", method="tf_idf")
    # train_model(algorithm="naive_bayes", method="tf_idf")
    # test_model(algorithm="naive_bayes", method="tf_idf")
    # train_model(algorithm="linear_regression", method="tf_idf")
    # test_model(algorithm="linear_regression", method="tf_idf")
    # train_model(algorithm="random_forest", method="tf_idf")
    # test_model(algorithm="random_forest", method="tf_idf")

    # create_results_chart()
    txt = "Fahrettin Koca yeni tedbirlere yönelik açıklamalarda bulundu."
    result = predict(txt, algorithm="svm", method="bag_of_words")
    print(f"Tahmin sonucu: {result}")
