from summarizer import summarizer
import trace
import argparse

tracer = trace.Trace(count=False, trace=True, ignoremods=(
    'argparse', 'numpy', 'os', 'pyrouge', 'nltk', 'sys', 're', 'bs4',
    'collections', 'itertools', 'string', 'posixpath', 'porter', 'punkt',
    'sre_compile', 'sre_parse',
    'genericpath', 'abc', '_weakrefset', 'fromnumeric', 'linalg', '_methods',
    'twodim_base', 'threading', 'platform', 'Rouge155', 'UserDict', 'ConfigParser',
    'subprocess', 'codecs', 'tempfile', 'random', '__init__', 'utils'))

parser = argparse.ArgumentParser(
    description="Multi-Document Text Summarizer.")
summarizer.parseArgs(parser)
options = parser.parse_args()


tracer.run('summarizer.run(options)')
