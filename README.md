Project Topic
=============

For this project, we propose focusing on the topic of single-document summarization (generation of new text summarizing assigned class readings, textbook chapters, etc.), with an extension to multi-document summarization to generate aggregations from new data sources (such as news sources). The latter part we expect to achieve as a reach-goal, and only if time proves sufficient.

Background Information
======================

We intend to explore the area of artificial intelligence known as [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) (NLP), and in particular, we intend to tackle the problem of document summarization. As implied by the references listed in this proposal, document summarization is an extremely important area in NLP. Application of automatic text summarization are evident, such as aggregating news sources, class readings, or business reports. Currently, such tasks are performed by assistants or aides, and while we do not expect our system to accurately summarize all documents, the literature shows that technical documents (such as annual business reports, textbook chapters, etc), are within the reach of AI. In our case, we will focus exclusively on material assigned to a typical college student (ie, textbook chapters, readings, research papers, etc).

System Overview
===============

The system we intend to build will take as input a document \(D\) of a particular length \(|D|\), where \(|D|\) is defined to be the number of words, and will generate a new document \(D'\) which summarizes the key points in the original documents. The new document will present factual information in a human-readable way. We will include a hard limit where \(|D'| \leq c|D|\) for some \(0 < c < 1\) yet to be determined. In general, we are hopeful that our system will be able to intelligently determine the best length of the summarization. We will maintain our focus on summarizing the document, therefore the system will take as input an arbitrary text file consisting of English words [1]. We hope that tackling the representation issue will require minimal amount of work, as we expect to focus on different models for summarizing text.

In more detail, the document will take as input a text file. The first step will consists of translating this document into a valid, machine-friendly representation. For a simple proposal, we consider using a [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) approach and determining word rankings (based on frequency of the words, ignoring [stop-words](https://en.wikipedia.org/wiki/Stop_words)). We also plan to look into more complicated, NLP-based approaches.

The next step is summarization. From what we’ve found in the literature, the most common, model approaches involve machine learning methods such as hidden markov models and neural nets. The survey paper by Das and Martins has a list of possible approaches, though we’ll focus on HMMs for the project and extend to other methods if time allows. In general, the focus will remain on machine learning methods for NLP, with an emphasis on document summarization. For additional methods, we’re interesting in learning more about neural nets (discussed in survey paper ), as well as naive bayes classifiers.

Extensions
==========

Multi-document summarization . We plan to only tackle single-document summarization, and if time allows, are hopeful to attempt an extension on our project that allows for news aggregation by summarizing multiple documents sources.

Neural networks, sentence compression. The survey paper lists a few additional methods that we’re interested in exploring, eventually providing quantitative evaluation feedback for each system we design. For example, summarization through sentence extraction appears promising .

Collaboration Plan
==================

We plan to explore multiple methods of text summarization, as discussed above. Given that the literature currently leans towards HMMs as good methods for understanding text (and that we’re most excited about HMMs), Kevin and I plan to collaborate concurrently on implementing a summarization system based on HMMs. The project will be open sourced and hosted on github. While empty, there current repository can be found here: <https://github.com/kandluis/document_summaries>

As for data collection, [Project Gutenberg](https://archive.org/details/gutenberg) appears to be a good source of text documents we can attempt to feed into our system.

Tools
=====

We expect the following tools to be useful when interpreting unstructured text.

-   [Python Natural Language ToolKit](http://www.nltk.org/)

-   [Stanford CoreNLP Tools](http://nlp.stanford.edu/software/corenlp.shtml)

[1] We do not expect to have the time to generalize to multiple languages, though from a quick search in the literature, multi-language summarization is an active area of research. It lies somewhat outside the scope of our project, however.
