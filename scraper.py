import requests
from bs4 import BeautifulSoup as bs
from deep_translator import GoogleTranslator as GT

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}

response = requests.get("https://old.reddit.com/r/espanol", headers = headers)

soup = bs(response.text, 'html.parser')

data = soup.find_all('a', class_="title")

translator = GT(source = 'es', target = 'en')

for datum in data:
    print("------------Original Text-----------")
    print(datum.text)
    print("-----------Translated text----------")
    print(translator.translate(datum.text))
