import re
import numpy as np
import pandas as pd
from string import punctuation, digits
from nltk.tokenize import word_tokenize


def get_stopwords():
    file = open("utils/stopwords.txt", "r", encoding="UTF-8")
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def clear_text(text, stop_words):
    try:
        translator = str.maketrans("", "", punctuation)
        text = text.translate(translator)
        translator = str.maketrans("", "", digits)
        text = text.translate(translator)
        text = word_tokenize(text.lower())
        text = [word for word in text if not word in stop_words and len(word) > 1]
        text = " ".join(text)
        pattern = r"[{}]".format(",.;’")
        text = re.sub(pattern, "", text)
        return text
    except:
        pass


def get_metadata(news):
    metadata = news.groupby(["class"]).size().to_frame("size")
    metadata.to_csv("data/metadata.csv")
    print("Verilerin metası oluşturuldu!")


def build_data():
    df = pd.read_csv("data/raw/data.csv")
    df2 = pd.DataFrame()
    df2["class"] = df["category"]

    stop_words = get_stopwords()
    df2["text"] = (df.title + " " + df.summary).apply(
        lambda x: clear_text(x, stop_words)
    )
    df2.dropna(subset=["text"], how="all", inplace=True)
    df2.to_csv("data/processed/data.csv", encoding="utf-8", index=False)
    print("Verilerin önişlemesi tamamlandı!")
    get_metadata(df2)


if __name__ == "__main__":
    build_data()
