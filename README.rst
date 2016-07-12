
====================================================
Topic modelling from EuroPython's list of abstracts
====================================================

This is the code to produce a list of topics from abstracts downloaded
from the conference website.

The different steps and corresponding modules are:

* **Web srapping** to retrieve the abstracts, based on `beautifulsoup4`,
  and `urllib2`.

  `joblib` is also useful for caching, to avoid multiple crawls of the
  websites and downloads.

  I could have asked access to a dump of the database for the organizers,
  but it was more fun to crawl.

* **Stemming**: trying to convert plural words to singular, using `NLTK`.

  Note that stemming is in general more sophisticated, and will convert
  words to their roots, such as 'organization' -> 'organ'. To have
  understandable word clouds, we want to keep more differentiation. Hence
  we add a custom layer to reduce the power of the stemmer.

* **Topic modelling** with `scikit-learn`.

  It's a 2 step process: first we convert the text data to a numerical
  representation, "vectorizing"; second we use a Non-negative Matrix
  Factorization to extract "topics" in these.

* **Word-cloud figures** with the `wordcloud` module.

* **Create a webpace** with the `tempita`.

___


This application beautifully combines multiple facets of the Python
ecosystem, from web tools to PyData.

