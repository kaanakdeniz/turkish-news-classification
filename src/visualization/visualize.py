import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from sklearn.metrics import accuracy_score, classification_report


def get_categories():
    metadata = pd.read_csv("data/metadata.csv")
    categories = metadata["class"].tolist()
    return categories


def get_categories_texts():
    df = pd.read_csv("data/processed/data.csv")
    categories = get_categories()
    categories_texts = []
    for category in categories:
        cat_text = {}
        cat_text["category"] = category
        cat_text["text"] = df.loc[df["class"] == category]["text"].str.cat(sep="")
        categories_texts.append(cat_text)

    return categories_texts


def create_wordclouds():
    df = pd.read_csv("data/processed/data.csv")
    categories = get_categories()
    categories_texts = get_categories_texts()
    for category in categories_texts:
        wordcloud = WordCloud(
            width=400,
            height=400,
            max_font_size=50,
            max_words=100,
            background_color="white",
        ).generate(category["text"])

        wordcloud.to_file(
            "reports/wordclouds/" + category["category"] + "_wordCloud.png"
        )

    print("Kelime bulutları oluşturuldu!")


def create_freq_chart():
    df = pd.read_csv("data/processed/data.csv")
    categories = get_categories()
    categories_texts = get_categories_texts()

    for category in categories_texts:

        tokens = word_tokenize(category["text"])
        frequency_dist = FreqDist(tokens)
        tempList = []

        for item, count in dict(frequency_dist).items():

            dt = {}
            dt["word"] = item
            dt["count"] = count
            tempList.append(dt)

        df2 = pd.DataFrame(tempList).sort_values("count", ascending=False)[0:20]
        ax = df2.plot.bar(
            x="word",
            y="count",
            rot=70,
            figsize=(10, 9),
            fontsize=11,
            title=category["category"].title() + " Haberleri Kelime Frekansları",
        )
        ax.set_xlabel("Kelimeler")
        ax.set_ylabel("Frekans")
        plt.savefig("reports/charts/data_charts/" + category["category"] + "_chart.png")
        plt.close()

    print("Frekans grafikleri oluşturuldu!")


def create_results_chart():
    df = pd.read_csv(("reports/outputs/test_results.csv"))

    df_tf = df[df["method"] == "tf_idf"]
    df_bow = df[df["method"] == "bag_of_words"]

    if df_tf.empty == False:

        ax = df_tf.plot.bar(
            x="algorithm",
            y="score",
            rot=50,
            figsize=(10, 9),
            fontsize=12,
            title="Test Accuracies with TF-IDF",
        )
        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Accuracy")
        plt.savefig("reports/charts/model_charts/test_accuracies_tfidf.png")
        plt.close()

    if df_bow.empty == False:
        ax = df_bow.plot.bar(
            x="algorithm",
            y="score",
            rot=50,
            figsize=(10, 9),
            fontsize=12,
            title="Test Accuracies with BoW",
        )
        ax.set_xlabel("Algorithms")
        ax.set_ylabel("Accuracy")
        plt.savefig("reports/charts/model_charts/test_accuracies_bow.png")
        plt.close()

    print("Test sonuçları grafiği oluşturuldu!")
