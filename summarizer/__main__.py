'''
Main entry point for our text summarization program.

Copyright, 2015.

Authors:
Luis Perez (luis.perez.live@gmail.com)
Kevin Eskici (keskici@college.harvar.edu)

For help in running this package, run:
    python -m summarizer --help
from the parent directory.
'''
from . import summarizer


def main():
    '''
    Main program
    '''

    parser = argparse.ArgumentParser(
        description="Multi-Document Text Summarizer.")
    summarizer.parseArgs(parser)
    options = parser.parse_args()
    summarizer.run(options)


if __name__ == '__main__':
    main()
