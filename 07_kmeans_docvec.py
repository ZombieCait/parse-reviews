import os, datetime, sys
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='kmeans.log', filemode='a')

def model_info(model):
    print('%s, vocab_size: %d, corpus_count: %d, dbow_words: %d\nsg: %d, hs: %d, sample: %g, vector_size: %d, iter: %d, min_count: %d' %
          (('DM' if model.dm else 'DBOW'),
           len(model.wv.vocab.values()),
           model.corpus_count,
           model.dbow_words,
           model.sg, model.hs, model.sample,
           model.vector_size, model.iter, model.min_count))


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
    start_time = datetime.datetime.now()

    limit_test = 200000
    n_clusters = 50
    batch_size = 100

    kmeans_file = os.path.join('docvec_kmeansEKVAR.pkl')
    check_file = 'clusters_d2v.csv'

    model_file = os.path.join('doc2vecEKVAR.model')
    # model_file = os.path.join('..','model','naznach.doc2vec.wl.model')
    test_docs_file = os.path.join('ekvairing 1m.txt')

    assert os.path.exists(model_file), 'Model file %s not exists' % model_file
    assert os.path.exists(test_docs_file), 'Test Doc file %s not exists' % test_docs_file

    if not os.path.exists(kmeans_file):
        print('Load Doc2Vec ... ', end='', flush=True)
        model = Doc2Vec.load(model_file)
        print('done')
        print(model)
        print('Vocab size:', len(model.wv.vocab.values()))
        print('Corpus count:', model.corpus_count)

        docvecs = model.docvecs
        del model

        print('Normalize ... ', end='', flush=True)
        docvecs.init_sims(replace=True)
        print('done')

        vectors = docvecs.doctag_syn0norm
        del docvecs
        print('Vectors shape:', vectors.shape)
        print('Total size: {} bytes'.format(str(sys.getsizeof(vectors))))

        print('Clustering with k-means (%d clusters, %d batch_size) ... ' %(n_clusters, batch_size), end='', flush=True)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=48, batch_size=batch_size)

        kmeans.fit(vectors)
        del vectors
        print('done')

        print('Save k-means model: %s ... ' % kmeans_file, end='', flush=True)
        joblib.dump(kmeans, kmeans_file)
        print('done')
    else:
        kmeans = joblib.load(kmeans_file)

    # Тестовый файл для просмотра кластеров
    print('Load Doc2Vec ... ', end='', flush=True)
    model = Doc2Vec.load(model_file)
    print('done')
    print(model)
    print('Vocab size:', len(model.wv.vocab.values()))
    print('Corpus count:', model.corpus_count)

    print('Start infering ... ')
    doc_vectors = []
    i = 0
    start_time2 = datetime.datetime.now()
    for doc in DocsIterator(test_docs_file, limit=limit_test):
        model.random.seed(1)
        doc_vectors.append(model.infer_vector(doc.words, steps=20, alpha=0.2, min_alpha=0.02))
        if i > 0 and i % 10000 == 0:
            print('Vectorize doc file: %s, %d items [%s]' % (test_docs_file, i,
                                                             str(datetime.datetime.now() - start_time2)[:-7]))
        i += 1
    print('done')
    del model

    clusters = kmeans.predict(normalize(doc_vectors))
    print(np.bincount(clusters))

    print('Save file for checking: %s ... ' % check_file, end='', flush=True)
    doc_df = pd.read_csv(test_docs_file, sep='|', index_col=None, header=None, encoding='cp1251')
    doc_df = doc_df[:len(doc_vectors)]
    doc_df.rename_axis({0:'id', 1:'text'}, axis=1, inplace=True)
    doc_df['cluster'] = clusters
    doc_df.set_index('id', inplace=True)
    doc_df = doc_df[['cluster','text']]
    doc_df.to_csv(check_file, index_label='id', sep=';')
    print('done')

    print('Total time: %s' % str(datetime.datetime.now() - start_time)[:-7])
