import re
import json
import requests
import sys
import pandas as pd

news_list = []

feed_link = "https://cloud.feedly.com/v3/streams/contents?streamId=feed/"
query = "&count=1000"

cleaner = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
news_stops = ["SON DAKİKA", "SON DAKİKA HABERİ", "Devamı için tıklayınız"]

with open("utils/rss_links.json") as json_file:
    rss_links = json.load(json_file)


def clear_news(text):
    text = re.sub(cleaner, "", text)
    text = re.sub(news_stops[0], "", text)
    text = re.sub(news_stops[1], "", text)
    text = re.sub(news_stops[2], "", text)
    return text


def make_dataset():

    i = 1
    for link in rss_links:
        sys.stdout.write(f"\r{i}/{len(rss_links)} Haberler çekiliyor.")
        i += 1

        url = feed_link + link["link"] + query
        try:
            page = requests.get(url)
            response = page.json()
            news_items = response["items"]
        except requests.exceptions.RequestException as e:
            print(e)

        for item in news_items:
            try:
                news = {}
                news["category"] = link["category"]
                news["title"] = clear_news(item["title"])
                news["summary"] = clear_news(item["summary"]["content"])
                news_list.append(news)
            except:
                continue

    df = pd.DataFrame(news_list)
    df = df.drop_duplicates(subset=["title"], keep=False)
    df.to_csv("data/raw/data.csv", index=False)

    print("\nVeri seti başarıyla oluşturuldu!")


if __name__ == "__main__":
    make_dataset()
