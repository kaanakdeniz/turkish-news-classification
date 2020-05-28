import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


def transform_data():
    news = pd.read_csv("data/processed/data.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        news["text"], news["class"], test_size=0.2, random_state=42
    )
    transformed_data = [
        {"x_train": x_train},
        {"x_test": x_test},
        {"y_train": y_train},
        {"y_test": y_test},
    ]

    for item in transformed_data:
        for key, value in item.items():
            filename = "data/interim/" + key + ".pk"
            with open(filename, "wb") as file:
                pickle.dump(value, file)
    print("Veriler işlendi!")


if __name__ == "__main__":
    transform_data()
