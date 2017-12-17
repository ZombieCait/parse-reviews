import os, datetime, logging
import warnings
import pandas as pd
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DocsIterator(object):
    def __init__(self, file, limit=None):
        self.file = file
        self.limit = limit

    def __iter__(self):
        with open(self.file) as f:
            for i, line in enumerate(f):
                doc_id, doc_str = line.rstrip().split('|')
                yield TaggedDocument(doc_str.split(), [doc_id])
                if self.limit is not None and (i + 1) >= self.limit:
                    break


if __name__ == '__main__':
    import sys

    start_time = datetime.datetime.now()

    model_file = os.path.join('doc2vecEKVAR.model')
    docs_file = os.path.join('ekvairing 1m.txt')

    documents = DocsIterator(docs_file, limit=None)

    model = Doc2Vec(documents, size=100, dm=0, min_count=100, iter=25, workers=1, seed=4, alpha=0.025, min_alpha=0.001)

    model.save(model_file)

    print('Total time: %s' % (datetime.datetime.now() - start_time))


