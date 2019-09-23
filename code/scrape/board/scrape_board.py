from bs4 import BeautifulSoup as bs
from requests import get 
import os

urls_path = './code/scrape/board/urls_board.txt'
time_class = 'article__time'
speaker_class = 'speaker'
text_class = 'article'
output_folder = 'data/fed_board'

def get_urls(): 
    with open(urls_path, 'r') as f: 
        all_urls = f.read() 
    return [u for u in all_urls.replace('\n','').split(',') if u != '']


def scrape_info(url): 
    raw = get(url)
    soup = bs(raw.text, 'html.parser')
    time = soup.find('p', class_ = time_class).get_text()
    speaker = soup.find('p', class_ = speaker_class).get_text()
    text = soup.find('div', id = text_class).get_text()
    return time, speaker, text


urls = get_urls()
fnum = 0
for url in urls: 
    if fnum % 50 == 0: 
        print(url)
    try: 
        article_time, speaker, text = scrape_info(url)
    except: 
        print('Error with {}'.format(url))
    fnum += 1
    fname = 'fomc_statement_{}.txt'.format(fnum)
    flocation = os.path.join(os.getcwd(), output_folder, fname)
    with open(flocation, 'w') as f: 
        all_txt = article_time + '|' + speaker + '|' + text
        f.write(all_txt)

