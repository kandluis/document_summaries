'''
Main modules for summarizer package.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvard.edu)
'''
import os
import traceback
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
    'multiplegeomprior': baselines.multipleGeomPrior,
    'frequency': baselines.wordFreqBaseline,
    'textrank': textrank.textRank,
    'modifiedtextrank': textrank.modifiedTextRank,
    'grasshopper':   grasshopper.run_grassHopper,
    'modifiedgrasshopper': grasshopper.run_modified_grasshopper
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
    parser.add_argument("-b", "--bytes", default=-1,
                        help="Byte length of output summary. All output summaries" +
                        " will be truncated at this length if written out to a file." +
                        "A value of -1 will avoid almost all truncation (except " +
                        "the last character). If you're setting this, you likely also " +
                        "want to set SUMMARY_LENGTH to some large value.")
    parser.add_argument("--rouge_folder", default="cs182_data/programs/RELEASE-1.5.5",
                        help="Folder Containing the ROUGE Perl Executables. " +
                        "It must be provided if ROUGE is to be used.")
    parser.add_argument("--sort_sents",  default="False",
                        help="Boolean parameter specifying whether sentences " +
                        "should be sorted or not.")


def processSummary(sort_sents, sentences, D, mapping):
    if sort_sents:
        sentences = sorted(sentences)
    return [D[mapping[i]] for i in sentences]


def createSummaries(sum_algo, abs_path, out_path, sort_sents, k=5, bytes=665, multiDocument=False):
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
        summary = processSummary(sort_sents, *sum_algo(D, k))

        # Write out the summary
        filepath = os.path.join(out_path, "SetSummary.{}.txt".format(setID))
        with open(filepath, 'w+') as out:
            res = "\n".join([s.strip() for s in summary])
            out.write(res[:bytes])
    else:
        for i in range(len(D)):
            summary = processSummary(sort_sents, sum_algo([D[i]], k))
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
    bytes = int(opts.bytes)
    sort_sents = opts.sort_sents.lower() == 'true'
    k = int(opts.summary_length)

    if opts.summarize.lower() == 'true':
        try:
            algorithm = argsToAlgo[opts.algorithm.lower()]
        except KeyError:
            raise Exception(
                "{} is not an available algorithm!".format(opts.algorithm))
    else:
        algorithm = opts.algorithm

    inputParams = "_sorted={}_k={}_bytes={}".format(sort_sents, k, bytes)
    outpath = None if base is None else os.path.join(
        base, opts.algorithm + inputParams)
    if opts.summarize.lower() == 'true':
        if base is None:
            print "\n".join([s.strip() for s in algorithm(
                [sys.stdin.readlines()], k)][:k])
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
                createSummaries(algorithm, inpath, outpath, sort_sents,
                                k=k, multiDocument=True)
            except Exception as e:
                print "Failed with {}".format(inpath)
                if debug:
                    print traceback.print_exc()

    # If rouge score is input, attempt to score the results with pyrouge
    # Currently only handles multiple documents!
    if base is not None and opts.rouge_score == 'True':
        import pyrouge
        # NOTE THAT WE MUST CONSTRUCT THE ARGUMENTS TO ROUGE OURSELF
        rouge_dir = os.path.abspath(opts.rouge_folder)
        options = [
            '-e', os.path.join(rouge_dir, 'data'),
            '-b', bytes,
            '-c', 95,
            '-n', 4,
            '-w', 1.2,
            '-a',
            '-f', 'A',
            '-p', 0.5,
            '-t', 0
        ]
        args = " ".join(map(str, options))

        r = pyrouge.Rouge155(rouge_dir=rouge_dir, rouge_args=args)
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
