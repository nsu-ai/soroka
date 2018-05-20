from googlesearch import search
from requests import get
from bs4 import BeautifulSoup


# https://github.com/MarioVilas/googlesearch
# for url in search('владимир жириновский', tld='ru', lang='ru', stop=10):
#     print(url)

url = 'http://www.vif2ne.org/vstrecha/forum/arhprint/20235'
url = 'https://boosters.pro/champ_15'
response = get(url)
print(response.text)
html_text = response.text

# html_text = html_text.decode("utf8")
# html_text = html_text.encode("ascii",'ignore')

html_soup = BeautifulSoup(html_text, 'html.parser')

# a_text = news_soup.find_all('p')

plain_text = html_soup.get_text()
print('+++++++++++++++++++++++')
print('+++++++++++++++++++++++')
print('+++++++++++++++++++++++')
print(plain_text)

page = html_soup.find('p').getText()
paragraphs = []
for p in html_soup.find_all(p):
    txt = p.getText().strip()
    if txt:
        paragraphs.append(txt)


# =================
import requests
from bs4 import BeautifulSoup


def recursiveUrl(url, link, depth):
    if depth == 2:
        return url
    else:
        print(link['href'])
        page = requests.get(url + link['href'])
        soup = BeautifulSoup(page.text, 'html.parser')
        newlink = soup.find('a')
        if len(newlink) == 0:
            return link
        else:
            return link, recursiveUrl(url, newlink, depth + 1)

def getLinks(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    links = soup.find_all('a')
    for link in links:
        links.append(recursiveUrl(url, link, 0))
    return links

links = getLinks("http://francaisauthentique.libsyn.com/")
print(links)

# https://stackoverflow.com/questions/46629681/how-to-find-recursively-all-links-from-a-webpage-with-beautifulsoup?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa