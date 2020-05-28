import joblib
import pandas as pd
import pickle
import csv
from os.path import exists
from sklearn.metrics import accuracy_score, classification_report


def get_test_data():

    with open("data/interim/x_test.pk", "rb") as file:
        x_test = pickle.load(file)
    with open("data/interim/y_test.pk", "rb") as file:
        y_test = pickle.load(file)

    return x_test, y_test


def save_accuracy(algorithm, method, score):
    fields = ["algorithm", "method", "score"]
    row = [algorithm, method, score]
    try:
        if exists("reports/outputs/test_results.csv") == False:
            with open(r"reports/outputs/test_results.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
                writer.writerow(row)
        else:
            with open(r"reports/outputs/test_results.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        print("Sonuçlar kaydedildi!")
    except:
        pass


def create_classification_report(y_test, y_pred, algorithm, method):
    try:
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(f"reports/classification_reports/{algorithm}_{method}.csv")
        print("Sınıflandırma raporu oluşturuldu!")
    except:
        pass

def test_model(algorithm, method):

    x_test, y_test = get_test_data()
    try:
        model = joblib.load("models/" + algorithm + "_" + method + ".joblib")

        with open("models/vectorizers/" + method + "_vectorizer.pk", "rb") as file:
            vectorizer = pickle.load(file)
            x_test = vectorizer.transform(x_test)
            predicted = model.predict(x_test)
            score = accuracy_score(predicted, y_test)
            save_accuracy(algorithm, method, score)
            create_classification_report(y_test, predicted, algorithm, method)
    except:
        print("Model veya vektör bulunamadı, lütfen önce modeli eğitin.")


if __name__ == "__main__":
    test_model(algorithm="svm", method="bag_of_words")
