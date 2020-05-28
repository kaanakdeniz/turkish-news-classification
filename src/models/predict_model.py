import joblib
import pickle
import pandas as pd


def predict(text, algorithm="svm", method="bag_of_words"):

    try:
        model = joblib.load("models/" + algorithm + "_" + method + ".joblib")

        with open("models/vectorizers/" + method + "_vectorizer.pk", "rb") as file:
            vectorizer = pickle.load(file)

        text = vectorizer.transform([text])

        prediction = model.predict(text)[0]

        return prediction
    except:
        print("Model veya vektör bulunamadı, lütfen önce modeli eğitin.")
