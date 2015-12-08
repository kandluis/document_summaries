'''
Main modules for summarizer package.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvar.edu)
'''
import os
import traceback
import pyrouge
import nltk
import argparse
import sys

from . import grasshopper
from . import baselines
from . import textrank

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Dictionary mapping algorithm commandline parameters to
# their respective algorithm classes.
argsToAlgo = {
    'baseline': baselines.baseline,
    'geomprior':   baselines.geomPriorBaseline,
    'firstgeomprior':   baselines.modifiedGeomPriorBaseline,
    'frequency':   baselines.wordFreqBaseline,
    'textrank': textrank.textRank,
    'grasshopper':   grasshopper.run_grassHopper
}


def parseArgs(parser):
    parser.add_argument("-d", "--data_dir", default=None,
                        help="Base Directory containing summarization documents" +
                        " and model summaries. Summarization documents should be" +
                        " contained in the docs/ subdirectory. See README for" +
                        " details. If no directory is provided, input is streamed" +
                        " from STDIN (or provided text file) and results are " +
                        "output to STDOUT. ROUGE analysis is not performed.")
    parser.add_argument("-a", "--algorithm", default="frequency",
                        help="Algorithm to use for summarization. Output" +
                        " summaries are saved under the DATA_DIR/ALGORITHM/" +
                        " directory if a data_dir parameter is provided." +
                        "Current options are {}".format(argsToAlgo.keys()))
    parser.add_argument("-s", "--rouge_score", default="False",
                        help="The parameter is ignored in the case where DATA_DIR " +
                        "is not set. Otherwise, if ROUGE_SCORE, then the model " +
                        "and system summaries are scored using the ROUGE metrics " +
                        "and results are printed to STDOUT.")
    parser.add_argument("--debug", default="False",
                        help="Prints helpful debugging information.")
    parser.add_argument("--summarize", default="True",
                        help="If true, performs the summarization using the " +
                        "specified ALGORITHM. Otherwise, does not summarize.")
    parser.add_argument("-k", "--summary_length", default=5,
                        help="Sentence length of output summary. Note that a summary" +
                        " might be truncated to be shorter than this length.")
    parser.add_argument("-b", "--bytes", default=665,
                        help="Byte length of output summary. All output summaries" +
                        " will be truncated at this length if written out to a file." +
                        "A value of -1 will avoid almost all truncation (except the last character).")


def createSummaries(sum_algo, abs_path, out_path, k=5, bytes=665, multiDocument=False):
    # Extracted from the input folder name
    setID = abs_path.split('/')[-1]

    # Stores DocumentIDs
    docIDs = []

    # Create document set that we wish to evaluate
    D = []
    for filename in os.listdir(abs_path):
        # Only take files that we've parsed before!
        tmp = filename.split('.')
        if tmp[0] == 'Parsed':
            docIDs.append(tmp[1])
            filepath = os.path.join(abs_path, filename)
            with open(filepath) as inputDoc:
                text = inputDoc.read().strip()
                sentences = tokenizer.tokenize(text)
                D.append(sentences)

    # Pass this to the algorithm which should return the summary as
    # a list of sentences.
    if multiDocument:
        summary = sum_algo(D, k, bytes)
        # Write out the summary
        filepath = os.path.join(out_path, "SetSummary.{}.txt".format(setID))
        with open(filepath, 'w+') as out:
            res = "\n".join([s.strip() for s in summary])
            out.write(res[:bytes])
    else:
        for i in range(len(D)):
            summary = sum_algo([D[i]], k)
            filepath = os.path.join(
                out_path, "Summary.{}.txt".format(docIDs[i]))
            with open(filepath, 'w+') as out:
                res = "\n".join([s.strip() for s in summary])
                out.write(res[:bytes])


def run(opts):
    '''
    Runs our summarization software based on user options.
    '''
    base = None if opts.data_dir is None else os.path.abspath(opts.data_dir)
    debug = opts.debug.lower() == 'true'

    if opts.summarize.lower() == 'true':
        try:
            algorithm = argsToAlgo[opts.algorithm.lower()]
        except KeyError:
            raise Exception(
                "{} is not an available algorithm!".format(opts.algorithm))
    else:
        algorithm = opts.algorithm

    outpath = None if base is None else os.path.join(base, opts.algorithm)
    if opts.summarize.lower() == 'true':
        k = int(opts.summary_length)
        bytes = int(opts.bytes)
        if base is None:
            print algorithm([sys.stdin.readlines()], k, bytes)
            return
        # Create directory if it does not exist
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        inbase = os.path.join(base, 'docs')
        folders = dirs = [d for d in os.listdir(
            inbase) if os.path.isdir(os.path.join(inbase, d))]
        for folder in folders:
            inpath = os.path.join(inbase, folder)
            try:
                createSummaries(algorithm, inpath, outpath,
                                k=k, bytes=bytes, multiDocument=True)
            except Exception as e:
                print "Failed with {}".format(inpath)
                if debug:
                    print traceback.print_exc()

    # If rouge score is input, attempt to score the results with pyrouge
    # Currently only handles multiple documents!
    if base is not None and opts.rouge_score == 'True':
        r = pyrouge.Rouge155(bytes=bytes)
        r.system_dir = outpath
        if debug:
            print "System Directory: {}.".format(r.system_dir)
        r.model_dir = os.path.join(base, 'model_multi')
        if debug:
            print "Model Directory: {}.".format(r.model_dir)
        r.system_filename_pattern = 'SetSummary.(\d+).txt'
        r.model_filename_pattern = 'SetSummary.#ID#.[A-Z].txt'

        output = r.convert_and_evaluate()

        print output
