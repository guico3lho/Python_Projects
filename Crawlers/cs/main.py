import time

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

import scrapy

class QuotesSpider(scrapy.Spider):
    name = 'QuotesSpider'
    start_urls = ['https://csgoempire.com/']

    def parse(self, response):
        quotes = response.xpath('//*[@id="page-scroll"]/div[1]/div[2]/div/div[7]/div/div[1]/div[2]/div[10]/div').get()
        for q in quotes:
            pass
def main():

    QuotesSpider()



if __name__ == '__main__':
    main()


#
# service = Service(ChromeDriverManager().install())
#
# navigator = webdriver.Chrome(service=service)
#
# navigator.get('https://csgoempire.com/')
#
# time.sleep(5)
# navigator.find_element('//*[@id="page-scroll"]/div[1]/div[2]/div/div[7]/div/div[1]/div[2]/div[10]/div').get()
#
# # //*[@id="page-scroll"]/div[1]/div[2]/div/div[7]/div/div[1]/div[2]/div[10]/div



# //div[@id='app']/div[@class="site-layout"]/div[@class="site-layout__main relative z-10"]/div

# //div[@id='app']/div[@class="site-layout"]/div[@class="site-layout__main relative z-10"]/div/div/div[@class="page-layout__inner"]/div[@class="page"]/div/div[@class="hidden lg:flex items-center justify-center mb-5"]/div/div[@class="flex flex-col lg:flex-row items-center mb-2 lg:mr-4 lg:mb-0"]/div[@class="flex relative h-24"]