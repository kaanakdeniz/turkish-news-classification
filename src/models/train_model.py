import pickle
from joblib import dump, load
from sklearn.feature_extraction.text import (
    TfidfTransformer,
    TfidfVectorizer,
    CountVectorizer,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier


def feature_extraction(data, method, max_features=50000):

    if method == "tf_idf":
        vectorizer = TfidfVectorizer()

        data = vectorizer.fit_transform(data)
        with open("models/vectorizers/tf_idf_vectorizer.pk", "wb") as file:
            pickle.dump(vectorizer, file)
        print("Vectorizer oluşturuldu ve kaydedildi!")

        return data

    if method == "bag_of_words":
        vectorizer = CountVectorizer(max_features=max_features)

        data = vectorizer.fit_transform(data)
        with open("models/vectorizers/bag_of_words_vectorizer.pk", "wb") as file:
            pickle.dump(vectorizer, file)
        print("Vectorizer oluşturuldu ve kaydedildi!")

        return data


def train_with_model(x_train, y_train, algorithm):

    if algorithm == "naive_bayes":
        nb = MultinomialNB()
        model = nb.fit(x_train, y_train)
        return model

    if algorithm == "linear_regression":
        lr = LogisticRegression(max_iter=1000, n_jobs=-1)
        model = lr.fit(x_train, y_train)
        return model

    if algorithm == "random_forest":
        rf = RandomForestClassifier(n_jobs=-1)
        model = rf.fit(x_train, y_train)
        return model

    if algorithm == "knn":
        knn = NearestCentroid()
        model = knn.fit(x_train, y_train)
        return model

    if algorithm == "svm":
        svm = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-3,
            random_state=42,
            max_iter=5,
            tol=None,
        )
        model = svm.fit(x_train, y_train)
        return model


def get_train_data():
    with open("data/interim/x_train.pk", "rb") as file:
        x_train = pickle.load(file)
    with open("data/interim/y_train.pk", "rb") as file:
        y_train = pickle.load(file)

    return x_train, y_train


def train_model(algorithm, method):
    x_train, y_train = get_train_data()
    x_train = feature_extraction(x_train, method=method)
    model = train_with_model(x_train, y_train, algorithm)
    dump(model, "models/" + algorithm + "_" + method + ".joblib")
    print(
        f"{algorithm.capitalize()} modeli {method.capitalize()} yöntemiyle eğitildi ve kaydedildi!"
    )


if __name__ == "__main__":
    train_model("svm", "tf_idf")
