import requests
response = requests.get("view-source:https://www.redditinc.com/")
print(response)

from bs4 import BeautifulSoup as bs
soup = bs(response.text, 'html.parser')
#print(soup.text)

data = soup.find_all('a', class_ = 'active')
print(data[0])