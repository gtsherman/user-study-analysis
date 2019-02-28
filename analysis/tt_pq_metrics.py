import argparse
import collections

import pyndri

from retrieval.core import Query, IndexWrapper, Qrels
from retrieval.scoring import jaccard_similarity, cosine_similarity, average_precision


def combine_vectors(*vectors):
    vector = collections.Counter()
    for v in vectors:
        vector += v
    return vector


def main():
    options = argparse.ArgumentParser()
    options.add_argument('topic_terms')
    options.add_argument('pseudo_queries')
    options.add_argument('index')
    args = options.parse_args()

    index = IndexWrapper(pyndri.Index(args.index))

    pseudo_query_terms = collections.defaultdict(collections.Counter)
    with open(args.pseudo_queries) as f:
        for line in f:
            docno, term, weight = line.strip().split(',')
            pseudo_query_terms[docno][term] = float(weight)

    pseudo_queries = {}
    for docno in pseudo_query_terms:
        pseudo_queries[docno] = Query(docno, vector=pseudo_query_terms[docno])

    topic_terms = collections.defaultdict(dict)
    with open(args.topic_terms) as f:
        for line in f:
            user, docno, _, term = line.strip().split(',')
            if docno not in topic_terms[user]:
                topic_terms[user][docno] = []
            topic_terms[user][docno].append(term)

    for user in topic_terms:
        for docno in topic_terms[user]:
            tts = topic_terms[user][docno]

            tt_query = Query(docno, vector=collections.Counter(tts))
            pseudo_query = pseudo_queries[docno]

            tt_results = index.query(tt_query, count=10)
            pseudo_results = index.query(pseudo_query, count=10)

            tt_result_docs = set([doc for doc, _ in tt_results])
            tt_result_docnos = set([doc.docno for doc in tt_result_docs])
            pseudo_result_docs = set([doc for doc, _ in pseudo_results])
            pseudo_result_docnos = set([doc.docno for doc in pseudo_result_docs])

            tt_pseudo_doc = combine_vectors(*[r.document_vector() for r in tt_result_docs])
            pseudo_pseudo_doc = combine_vectors(*[r.document_vector() for r in pseudo_result_docs])

            tt_qrels = Qrels()
            tt_qrels._qrels[docno] = set(tts)

            results_jaccard = jaccard_similarity(tt_result_docnos, pseudo_result_docnos)
            pseudo_recall = len(tt_result_docnos & pseudo_result_docnos) / len(tt_result_docnos)
            cosine = cosine_similarity(tt_pseudo_doc, pseudo_pseudo_doc)
            pseudo_ap = average_precision(docno, sorted(pseudo_query.vector.keys(), key=lambda k:
                                                        pseudo_query.vector[k], reverse=True), tt_qrels)

            print(user, docno, results_jaccard, pseudo_recall, cosine, pseudo_ap, sep=',')


if __name__ == '__main__':
    main()
