import scrapy

# class QuotesSpider(scrapy.Spider):
#     name = 'QuotesSpider'
#     start_urls = ['http://quotes.toscrape.com/']
#
#     def parse(self, response):
#         for quote in response.css('div.quote'):
#             yield {
#                 'text': quote.css('span.text::text').get(),
#                 'author': quote.css('small.author::text').get(),
#                 'tags': quote.css('div.tags a.tag::text').getall(),
#             }
#         next_page = response.css('li.next a::attr(href)').get()
#         if next_page is not None:
#             yield response.follow(next_page, self.parse)
#

class QuotesSpider(scrapy.Spider):
    name = 'QuotesSpider'
    start_urls = ['http://quotes.toscrape.com/']

    def parse(self, response):
        quotes = response.xpath('//div[@class="quote"]')
        for q in quotes:
            yield {
                'text': q.xpath('.//span[@class="text"]/text()').get(),
                'author': q.xpath('.//small/text()').get(),
            }
def main():

    QuotesSpider()



if __name__ == '__main__':
    main()

# response.xpath("*//div/span[@class='text']/text()").getall()[0]
