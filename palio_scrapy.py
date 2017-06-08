# -*- coding: utf-8 -*-

import scrapy

class PalioSpider(scrapy.Spider):
	name = "palio"

	start_urls = ['http://carros.mercadolivre.com.br/carros-e-caminhonetes/fiat/palio/_DisplayType_LF']

	def parse(self, response):
		# Follow links to palios
		for href in response.css('div.item__info-container div.item__info a.item__info-title::attr(href)'):
			yield response.follow(href, self.parse_palio)

		# Follow pagination links
		for href in response.css('ul.pagination.stack.u-clearfix li.pagination__next a::attr(href)'):
			yield response.follow(href, self.parse)

	def parse_palio(self, response):
		def extract_css(query):
			return response.css(query).extract()

		def extract_css_first(query):
			return response.css(query).extract_first()

		yield {
		    'properties': extract_css('div.card-section dl.ch-box.attribute-group.clear-floats dd.attribute-value.text-light::text'),
		    'price': extract_css_first('article.vip-price.ch-price strong::text').replace("R$ ", "")
		}