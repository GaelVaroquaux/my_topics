
all: html

html: topics_extraction.py
	python topics_extraction.py

install: html
	python github-pages-publish/github-pages-publish . out/
	git push origin gh-pages
