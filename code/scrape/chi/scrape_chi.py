from bs4 import BeautifulSoup as bs
from requests import get 
import os

##################################################################################
# NOTE: call from base directory fed-statements
##################################################################################

urls_path = './code/scrape/chi/urls_chi.txt'
time_class = 'cfedDetail__lastUpdated'
text_class = 'cfedContent__text'
speaker_name = 'Charlie Evans'

fname_template = 'chi_statement_{}.txt'
output_folder = 'data\\fed_chi'
data_path = os.path.join(os.getcwd(), output_folder)

def get_urls(): 
    with open(urls_path, 'r') as f: 
        all_urls = f.read() 
    return [u for u in all_urls.replace('\n','').split(',') if u != '']


def scrape_info(url): 
    raw = get(url)
    soup = bs(raw.text, 'html.parser')
    time = soup.find('div', class_ = time_class).get_text()
    try: 
        time = time.split(':')[1].strip()
    except: 
        print('Time is {}\nUnable to strip time from format "Last Updated: ..."'.format(time))
    speaker = speaker_name
    text = soup.find('div', class_ = text_class).get_text()
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
    fname = fname_template.format(fnum)
    
    # call from base directory fed-statements
    flocation = os.path.join(data_path, fname)
    if not os.path.exists(data_path):
        print('Making data directory')
        os.mkdir(data_path)

    with open(flocation, 'w') as f: 
        all_txt = article_time + '|' + speaker + '|' + text
        f.write(all_txt)
