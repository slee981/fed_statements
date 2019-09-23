from selenium import webdriver 
import time

ALL_URLS = []
current_page = 1
max_pages = 39

url = 'https://www.federalreserve.gov/newsevents/speeches.htm'
class_name = 'ng-binding'
page_link_class_name = 'pagination-page'
path_to_chromedriver = 'C:\\Users\\g1sml02\\Downloads\\chromedriver.exe'
final_urls_fname = 'urls_board.txt'

driver = None


# get urls from page
def get_urls():
    global driver
    urls_section = driver.find_elements_by_class_name(class_name)
    for u in urls_section: 
        href = u.get_attribute('href')
        if href != None and len(href) > 1: 
            ALL_URLS.append(href)


def next_page():
    global current_page, driver
    page_nav = driver.find_elements_by_class_name(class_name)
    numbers_checked = 0
    # click on the next page number
    for page in page_nav: 
        txt = page.text
        try: 
            num = int(txt)
            numbers_checked += 1
            if num == current_page: 
                page.click()
                print('Scraped page {}'.format(num))
                current_page += 1
                return
        except: 
            # if we didn't find one, we need to click on the '...'
            if txt == '...' and numbers_checked > 1: 
                page.click()
                current_page += 1
                return


def write(lst, fname):
    with open(fname, 'a') as f: 
        for ele in lst: 
            row = ele + ',\n'
            f.write(row)


def main(): 
    global current_page, max_pages, driver

    # load initial page 
    driver = webdriver.Chrome(path_to_chromedriver)
    driver.get(url)

    # while current page < max pages
    while (current_page <= max_pages):
        
        # get all urls
        get_urls()

        # click on next page OR the '...' AND THEN click on your number
        next_page()

    write(ALL_URLS, final_urls_fname)
    driver.quit()


if __name__ == '__main__': 
    main()