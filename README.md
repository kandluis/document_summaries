SampleRuns
=========
To see how the program runs, you can execute it with the included DUC 2004 Data. First, make sure the required packages listed under set-up are installed.

For a quick sample run, you can run this command from the root directory:

```
python -m summarizer --data_dir=sample_rouge_data --algorithm=grasshopper --rouge_score=True --summarize=True
```

Feel free to change the --algorithm flag to one of the other available and implemented algorithms. Furthermore, if you do not have ROUGE install, the --rouge_score flag must be set to false.


Set-up
=====
Required packages for correct parsing of results include:
  - numpy (required)
  - nltk (required)
  - pyrouge (optional)

Note that nltk data must be downloaded using
```
nltk.download()
```

Note that you must also set-up pyrouge to work with your install of ROUGE.

If running ROUGE metrics, a valid install of ROUGE is required. For somewhat ambiguous instructions on how to do this (we highly recommend you don't attempt to install ROUGE...), you can attempt to follow the [PyRouge instructions](https://pypi.python.org/pypi/pyrouge/0.1.2) or [StackOverflow](http://stackoverflow.com/questions/28941839/how-to-install-rouge-in-ubuntu) (for Ubuntu systems).

You can see more information how to run the program with the command:
```
python -m summarizer --help
```

Program Data
============
The data used for testing the system can be found [here](https://www.dropbox.com/sh/dzmzh5nwe1i68ra/AABYPkOj6lXZln5I6tEDjpAna?dl=0).

For more data requests, please email Luis Perez (luis.perez.live@gmail.com).

Program Arguments
==================
-d, --data_dir, default=None
  Base Directory containing summarization documents and model summaries. Summarization documents should be contained in the docs/ subdirectory. All documents need to be pre-processed so that the file name syntax is given by /docs/SETID/Parsed.DOCID.txt. The format is what ROUGE expects. It is possible to avoid this requirement if the system is asked to ignore ROUGE scores (and therefore does not calculate them). A utility is included in the utils.py directory: cleanOriginalDocs(docs). Given a directory directly from the DUC 2004 conference, this utility will convert the output into the input expected by the system.

  If ROUGE is not to be used, then the system will accept a directory which follows the same overall hierarchy (ie, document sets should be contained in sub-folders), but does not necessarily match the required file patterns for Rouge. However, all documents must be prefixed with "Parsed.", as this is the input format that the system expects!

  If no directory is provided, input is streamed from STDIN (or provided text file) and results are output to STDOUT. ROUGE analysis is not performed. This is likely to be the main use for the system, as it will allow anyone to summarize arbitrary text. Note that the text is split into sentences using the NLTK Punkt tokenizer.

-a, --algorithm, default="frequency"
  Algorithm to use for summarization. Output summaries are saved under the DATA_DIR/ALGORITHM/ directory if a data_dir parameter is provided. Otherwise, results are simply output to STDOUT and must be saved by the user. For an updated list of the available algorithms, pass --help flag.

-s, --rouge_score, default=False
  The parameter is ignored in the case where DATA_DIR is not set. Otherwise, if ROUGE_SCORE, then the model and system summaries are scored using the ROUGE metrics and results are printed to STDOUT.


--debug, default=False
  Prints helpful debugging information. If the program crashes, setting this flag to true is useful.

--summarize, default=True
  If true, performs the summarization using the specified ALGORITHM. Otherwise, does not summarize and instead the ALGORITHM parameter simply becomes the output directory where the summarized results should be contained! Setting this to False is helpful if you have already summarized a set of documents previously (or with another system), and wish to simply calculate the ROUGE score for comparison. It is common to have --summarize=False and --rouge_score=True

-k, --summary_length, default=5
  Sentence length of output summary. Note that a summary might be truncated to be shorter than this length. Currently, summaries are truncated anytime they are output to a file. We work under the assumption that these summaries will then be passed into ROUGE for evaluations.

-b, --bytes, default=665
  Byte length of output summary. All output summaries will be truncated at this length if written out to a file. A value of -1 will avoid almost all truncation (except the last character).

Multi-Document Text Summarization
=======
[Kevin Eskici](mailto:keskici@college.harvard.edu) and [Luis A.
Perez](mailto:luisperez@college.harvard.edu), Harvard University


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


Problem Statement
=================

The problem has been redefined slightly, as we’re now taking an extractive approach to textual summarization. We will be using the NLTK Python Library ([NLTK API](http://www.nltk.org/api/nltk.html)) along with the accompanying texts ([Link to Corpus Data](http://www.nltk.org/nltk_data/)). The reason for this decision is that the library is readily available and provides us with many natural language processing features that we would rather not implement.

Baseline
========

An important step in the project, on which we’re still working on but are hoping to have done soon is to establish a baseline against which we can compare the results of more advanced algorithms. Our proposed baseline procedure is below.

We begin our approach by first defining the function \(|\cdot|_s : \mathbb{D} \to \mathbb{N}^+\) as the length of the document \(D\) in terms of number of sentences (similarly \(|\cdot|_w\) for the number of words), where \(\mathbb{D}\) is the space of documents. Let \(\mathbb{S}\) be the state of possible sentences, and \(\mathbb{W}\) the state of our vocabulary words. Then define an extractive summary, \(\mathcal{S}\), for a documented \(D\),as an ordered collection of the most “important” sentences \(\{s_i\}_{i=1}^k\). We have that \(k\) is a parameter selected by the user with \(1 \leq k \leq |D|\).

With the above set-up, we now present our proposed baseline by making note of some intuitive features about sentences.

-   We note that the importance of a sentence, defined as a function \(i_s: \mathbb{S} \to [0,1]\). The importance of a sentence, naively, depends on the importance of the words in the sentence, so we can imagine that \(i = f \circ i_w \circ g\) where \(i_w: \mathbb{W} \to [0,1]\) is the importance of a single word. Note that all of the importance functions have codomain that is normalized, though they are not strictly probability distributions.

-   As per discussion in the literature , the importance of a sentence also depends on the location of the sentence relative to the body of work. Very naively, we modify \(i_s' = \mathbb{S} \times \mathbb{N} \to [0,1]\) as the extension of \(i_s\) to the space containing indexed locations.

-   As for the importance of a word, we simply measure the number of times the word occurs in our body of work. \[i_w(w) = \frac{\text{count}_D(w)}{|D|_w}\]

-   Putting the above ideas together, we define the importance of a sentence as \[i'_s(s,i) = \sum_{w \in s} i_w(w)\] where we normalize the results after calculating them by dividing by: \(\sum_{(s',i') \in D - \{w\}} i'_s(s',i')\).

-   The last idea involves a further modification to the calculation of the sentence score. Suppose we have selected the set of sentences \(\{s_i\}_{i=1}^m\) for our summary, and are looking to select a further sentence \(s_{m+1}\). In order to calculate the score of the sentences, we consider the new word space \(\mathbb{W'} = \mathbb{W} \setminus A\) where \(A = \bigcup_{i=1}^k A_i\) with \(A_i = \{w \mid w \in s_i\}\). The idea here being that we do not want to double count words that have already been selected!

-   The final item is that we break ties at random.

Current Work
============

Challenges
----------

We’ve struggled with finding a good source of training data for our model, as has been highlighted above. In the weeks up to this update, we’ve spent most of our time researching current models. We’ve actually diverged from our original plan of utilizing an HMM due to the lack-luster results we’ve found in other papers (after further research). However, we’ve stayed true to the idea of making use of Markov Models by researching the Grasshopper algorithm (see Section [sec:grasshopper]). Additionally, the lack of good training data (it’s surprising that papers don’t provide their source code/data along with the paper itself) has led us to instead attempt an unsupervised approach to the problem.

Implementation of Baselines
---------------------------

The baselines are not yet implemented, as of the writing of this update.

Implementation of Grasshopper Algorithm
---------------------------------------

The Grasshopper algorithm is described in the has been implemented. We now present a brief overview of this algorithm so the discussion in the Section [sec:future<sub>w</sub>ork] section makes sense.

-   Grasshopper creates a Markov model of the given document. Each sentence is represented as a node in the graph, with edges to all other sentence with weight \(w_{ij}\) which is dependent on the cosine similarity between sentence \(i\) and sentence \(j\). We plan to use TF-IDF vectors for each sentence. Note that a sparce graph is enforced by using a \(0.1\) thresh-hold for any edge weight.

-   We further modify \(w_{ij}\) by taking into account the prior \(r\), over the sentence importance, which serves to weigh some edges more heavily than others.

-   Grasshopper then finds the steady state of this Markov model.

-   From the steady state, the state with the largest probability distribution (\(s_{max}\)) is selected.

-   The graph is then modified to change the state \(s_{max}\) into an absorption state.

-   Repeat the the above steps (excluding the first) until a total of \(k\) sentences are extracted, as specified by the user.

The algorithm is relatively simple. Example output for ca19 from the Brown Corpus in NLTK is now presented. We use a weight parameter \(\lambda\) on the prior of \(0\), along with simple word-count sentence vectors. Note that we’ve yet to implement the removal of stopwords, etc., and the below is completely preliminary in results.

> Howard E. Simpson , the railroad’s president , said , “ A drastic decline in freight loading due principally to the severe slump in the movement of heavy goods has necessitated this regrettable action ” . Since the railroad cannot reduce the salary of individual union members under contract , it must accomplish its payroll reduction by placing some of the men on furlough , a B. & O. spokesman said . The proposal was made by Dr. David S. Jenkins after he and Mrs. D. Ellwood Williams , Jr. , a board member and long-time critic of the superintendent , argued for about fifteen minutes at this week’s meeting . Cites discrepancies Soon after 10 A.M. , when police reached the 1-1/2-story brick home in the Franklin Manor section , 15 miles south of here on the bay , in response to a call from the Dresbach’s other son , Lee , 14 , they found Mrs. Dresbach’s body on the first-floor bedroom floor .

Implementation of TextRank
--------------------------

Another approach we are exploring is using the TextRank algorithm described in . At this time we have a simple implementation running for single document extractive summarization (example below). Key to the results is the similarity function used to compute edge weights between sentences. For starters we used the one described in the paper where the similarity score was a function of sentence lengths and the number of words they had in common (filtering out non Nouns, Adjectives, and Verbs).

Example output for ca19 from the Brown Corpus in NLTK:

> The Baltimore and Ohio Railroad announced yesterday it would reduce the total amount of its payroll by 10 per cent through salary cuts and lay-offs effective at 12:01 A.M. next Saturday. Howard E. Simpson , the railroad’s president, said,“ A drastic decline in freight loading due principally to the severe slump in the movement of heavy goods has necessitated this regrettable action”.A thug struck a cab driver in the face with a pistol last night after robbing him of $18 at Franklin and Mount Streets . The victim , Norman B. Wiley , 38 , of the 900 block North Charles Street , was treated for cuts at Franklin Square Hospital after the robbery . A baby was burned to death and two other children were seriously injured last night in a fire which damaged their one-room Anne Arundel county home.

Future Work
-----------

The first step is to implement tests. While some of our current results appear promising, we plan on using ROUGE along with a yet to be discovered corpus to help us compare our summaries to those of humans (or generated by other machines?).

For the Grasshopper algorithm, once we have found some training data, we plan to utilize the training set to optimize our \(\lambda, \alpha\) parameters to the model. These optimizations can be done through Bayesian Optimization using the [Spearmint](https://github.com/HIPS/Spearmint). Following the suggestions in , we also plan to to generalize the grasshopper algorithm to consider “partial absorbtion” rather than full absorption, allowing for more variation in our sentence selection. Additionally, when learning the parameters, we plan to utilize different forms of priors \(r\) based on textual analysis of the documents. We will compare these results to those that we’ve achieved thus far and select the most optimal model. Lastly, we might look into changes the \(0.1\) threshold used above to enforce sparsity.

For TextRank, we believe results can be greatly increased by exploring different similarity functions. Additionally the algorithm as presented in the paper does not seem well suited for selecting orthogonal sentences in a text, so we may explore variants that take this into account. One idea would be to run the algorithm to get our top sentence, and then rerun filtering out keywords in that sentence, and repeating until the desired number of sentences has been extracted.

Before optimizing these algorithms and extending them to multi document summerization, our focus will be to implement tests as described above so we can track results as we go.

