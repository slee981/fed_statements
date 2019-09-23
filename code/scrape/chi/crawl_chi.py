from requests import get
from bs4 import BeautifulSoup as bs 

# url template
url_template = 'https://www.chicagofed.org/publications/speeches/{}/index'
url_base = 'https://www.chicagofed.org'
years_available = range(2012, 2020)    # <- stop is exclusive 
urls_file = 'urls_chi.txt'
dates_url_file = 'dates_url_chi.txt'

def write(lst, fname, lst_of_lsts=False):
    with open(fname, 'a') as f: 
        for ele in lst: 
            if not lst_of_lsts: 
                row = ele + ',\n'
                f.write(row)
            else: 
                for e in ele: 
                    row = str(e) + ','
                    f.write(row)
                f.write('\n')

for year in years_available: 
    speech_links = []
    speech_dates = []
    url = url_template.format(year)
    r = get(url)
    soup = bs(r.text, 'html.parser')
    content = soup.find('div', class_='cfedContent__text')
    speeches = content.find_all('p')
    for speech in speeches: 

        info = speech.get_text()
        
        # get time
        info = info.split('\n')
        title = info[0]
        other = info[-1]
        try: 
            date, other = other.split('|')
        except: 
            date = other

        # only keep the relevant links
        for link in speech.find_all('a'): 
            l = url_base + link['href']
            if ('~/media' not in l) and 'publication' in l: 
                speech_info = (date, l)  

                speech_links.append(l)      
                speech_dates.append(speech_info)

    write(speech_dates, dates_url_file, lst_of_lsts=True)
    write(speech_links, urls_file)