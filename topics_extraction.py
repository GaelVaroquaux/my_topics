OUTPUT_DIR = "ep16"

###############################################################################
# Download the pages

# Beautiful soup, for webscraping
import bs4
import urllib2

import joblib

mem = joblib.Memory(cachedir='cache')

def get_list_of_talks():
    all_talks_urls = {}

    main_page = urllib2.urlopen(
            'https://ep2016.europython.eu/p3/schedule/ep2016/list/')
    tree = bs4.BeautifulSoup(main_page.read())

    rows = tree.find_all(name='td', attrs={'class': 'event'})

    for row in rows:
        divs = row.find_all(name='div', attrs={'class': 'name'})
        for div in divs:
            link = div.find_next(name='a')
            url = 'https://ep2016.europython.eu/' + link.attrs['href']
            title = link.get_text()
            if title:
                all_talks_urls[title] = url
                break

    return all_talks_urls


def grab_talk_description(talk_url):
    page = urllib2.urlopen(talk_url)
    tree = bs4.BeautifulSoup(page.read())

    # First extract the content
    content = tree.find_all(name='div', attrs={'class': 'cms'})[0].get_text()

    # Second grab the tags
    tag_div = tree.find_all(name='div', attrs={'class': 'all-tags'})
    if tag_div:
        tags = [t.get_text()
                for t in tag_div[0].find_all(name='span',
                                             attrs={'class': 'tag'})]
    else:
        tags = []

    return content, tags


all_talks_urls = mem.cache(get_list_of_talks)()

all_talks_description = {}
all_talks_document = {}

for title, url in all_talks_urls.items():
    content, tags = mem.cache(grab_talk_description)(url)
    all_talks_description[title] = content
    # Add the tags repeated 3 times to the content, to give them more weight
    all_talks_document[title] = '%s %s' % (content, ' '.join(3 * tags))


# Make a list of documents (sort for reproducibility)
documents = [d for t, d in sorted(all_talks_document.items())]


###############################################################################
# Stemming: converting words to a canonical form. Here we only worry
# about plural

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")

PROTECTED_WORDS = ['pandas', 'itertools']

def no_plural_stemmer(word):
    """ A stemmer that tries to apply only on plural. The goal is to keep
        the readability of the words.
    """
    word = word.lower()
    if word.endswith('s') and not (word in PROTECTED_WORDS
                                   or word.endswith('sis')):
        stemmed_word = stemmer.stem(word)
        if len(stemmed_word) == len(word) - 1:
            word = stemmed_word
    return word


###############################################################################
# Learn the topic model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


n_features = 1000
n_topics = 10

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (no_plural_stemmer(w) for w in analyzer(doc))

# We use a few heuristics to filter out useless terms early on: the posts
# are stripped of headers, footers and quoted replies, and common English
# words, words occurring in only one document or in at least 95% of the
# documents are removed.

# Use tf-idf features for NMF.
tfidf_vectorizer = StemmedTfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)

# Fit the NMF model
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

feature_names = tfidf_vectorizer.get_feature_names()

doc_loadings = nmf.transform(tfidf)

###############################################################################
# Plot word-cloud figures for each topic
import os
from wordcloud import WordCloud
import itertools

def my_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    """ hue in the 128-255 range given by size, with saturation 90% and
    lumination 20%"""
    return "hsl(%d, 90%%, 20%%)" % (110 + 3 * font_size)


# First create an ellipse mask
import numpy as np
x, y = np.ogrid[-1:1:250j, -1:1:450j]
mask = (255 * ((x ** 2 + y ** 2) > 1)).astype(int)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Generate a word cloud image using frequencies assign to the terms
for topic_idx, topic in enumerate(nmf.components_):
    freq_cloud = WordCloud(max_font_size=40, relative_scaling=0.5,
                           #background_color=None, mode="RGBA",
                           background_color='white', mode="RGBA",
                           mask=mask, color_func=my_color_func,
                           scale=1.5)
    frequencies = [(w, f)
                   for w, f in itertools.izip(feature_names, topic)
                   if f != 0]
    freq_cloud.generate_from_frequencies(frequencies)
    freq_cloud.to_file(os.path.join(OUTPUT_DIR, 'topic_%02i.png' % topic_idx))


###############################################################################
# Output an HTML file

titles_and_urls = sorted(all_talks_urls.items())

import tempita

# First create the information that will go in the file
topics = list()
for topic, loading in itertools.izip(nmf.components_, doc_loadings.T):
    frequencies = [(f, w)
                   for f, w in itertools.izip(topic, feature_names)
                   if f != 0]
    frequencies.sort(reverse=True)
    titles = [(l, t)
                   for l, t in itertools.izip(loading, titles_and_urls)
                   if l != 0]
    titles.sort(reverse=True)
    talks = [tempita.bunch(title=t[0], url=t[1],
              description=(all_talks_description[t[0]]
               if len(all_talks_description[t[0]].strip()) > 1
               else ""))
             for l, t in titles]
    topic_desc = tempita.bunch(first_word=frequencies[0][1],
                               second_word=frequencies[1][1],
                               talks=talks[:10])
    topics.append(topic_desc)

template = tempita.HTMLTemplate.from_filename('index_template.html')

html = template.substitute(topics=topics)
open(os.path.join(OUTPUT_DIR, 'index.html'), 'w').write(html)

