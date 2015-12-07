'''
Main entry point for our text summarization program.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvar.edu)

For full operation, run as follows:
    data_dir='/home/luis/Dropbox/OnlineDocuments/HarvardSchoolWork/Fall2015/cs182/project/cs182_data/rouge_data/'

'''
from optparse import OptionParser
from . import grasshopper
from . import baselines

# Dictionary mapping algorithm commandline parameters to
# their respective algorithm classes.
argsToAlgo = {
    'grasshopper':   grasshopper.run_grassHopper,
    'geomPrior':   baselines.geomPriorBaseline,
    'firstGeomPrior':   baselines.modifiedGeomPriorBaseline,
    'frequency':   baselines.wordFreqBaseline
}


def createSummaries(sum_algo, abs_path, out_path, k=5, multiDocument=False):
    # Extracted from the input folder name
    setID = abs_path.split('/')[-1]

    # Stores DocumentIDs
    docIDs = []

    # Create document set that we wish to evaluate
    D = []
    for filename in os.listdir(abs_path):
        # Only take files that we've parsed before!
        tmp = filename.split('.')
        if tmp[0] == 'parsed':
            docIDs.append(tmp[1])
            filepath = os.path.join(abs_path, filename)
            with open(filepath) as inputDoc:
                D.append(inputDoc.readlines())

    # Pass this to the algorithm which should return the summary as
    # a list of sentences.
    if multiDocument:
        summary = sum_algo(D, k)
        # Write out the summary
        filepath = os.path.join(out_path, "SetSummary.{}.txt".format(setID))
        with open(filepath, 'w+') as out:
            for s in summary:
                out.write("{}.\n".format(s.strip()))
    else:
        for i in range(len(D)):
            summary = sum_algo([D[i]], k)
            filepath = os.path.join(
                out_path, "Summary.{}.txt".format(docIDs[i]))
            with open(filepath, 'w+') as out:
                for s in summary:
                    out.write("{}.\n".format(s.strip()))


def parseArgs(parser):
    parser.add_option("-d", "--data_dir", default=None,
                      help="Base Directory Containing Rouge Data")


def run(opts):
    '''
    Runs our summarization software based on user options.
    '''
    base = opts.data_dir
    outpath = base + 'grass_out'
    inbase = base + 'docs/'
    for folder in os.listdir(inbase):
        inpath = inbase + folder
        try:
            createSummaries(grassHopper, inpath, outpath, multiDocument=True)
        except:
            print "Failed with {}".format(inpath)


def main():
    '''
    Main program
    '''

    parser = OptionParser()
    getArgs(parser)
    options, args = parser.parse_args()
    run(options)


if __name__ == '__main__':
    main()
