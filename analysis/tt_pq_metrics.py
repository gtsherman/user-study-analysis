import argparse
import collections
import math
import random
import sys

import pyndri

from retrieval.core import Query, IndexWrapper, Qrels
from retrieval.scoring import jaccard_similarity, cosine_similarity, average_precision, recall, build_vocab, \
    DirichletTermScorer


def combine_vectors(*vectors):
    vector = collections.Counter()
    for v in vectors:
        vector += v
    return vector


def main():
    options = argparse.ArgumentParser()
    # options.add_argument('topic_terms')
    options.add_argument('pseudo_queries')
    options.add_argument('index')
    options.add_argument('-n', '--num-results', type=int, default=10)
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

    # topic_terms[user][doc][term][weight]
    topic_terms = collections.defaultdict(lambda: collections.defaultdict(dict))
    # with open(args.topic_terms) as f:
    for line in sys.stdin:
        parts = line.strip().split(',')
        user, docno = parts[0], parts[1]
        terms = parts[3:]
        topic_terms[user][docno] = terms

    scorer = DirichletTermScorer(index)
    for user in topic_terms:
        for docno in topic_terms[user]:
            tts = topic_terms[user][docno]

            pseudo_query = pseudo_queries[docno]
            # as a test, let's do all the same comparisons against a random pseudo-query
            # pseudo_query = pseudo_queries[random.choice(list(pseudo_queries.keys()))]

            tt_vector = collections.Counter(tts)
            tt_query = Query(docno, vector=tt_vector)

            tt_results = index.query(tt_query, count=args.num_results)
            pseudo_results = index.query(pseudo_query, count=args.num_results)

            tt_result_docs = set([doc for doc, _ in tt_results])
            tt_result_docnos = set([doc.docno for doc in tt_result_docs])
            pseudo_result_docs = set([doc for doc, _ in pseudo_results])
            pseudo_result_docnos = set([doc.docno for doc in pseudo_result_docs])

            # tt_pseudo_doc = combine_vectors(*[r.document_vector() for r in tt_result_docs])
            # pseudo_pseudo_doc = combine_vectors(*[r.document_vector() for r in pseudo_result_docs])

            def normalize_results_scores(results):
                total = sum([score for _, score in results])
                return [(doc, score / total) for doc, score in results]

            tt_vocab = build_vocab(*[r.document_vector() for r in tt_result_docs])
            pseudo_vocab = build_vocab(*[r.document_vector() for r in pseudo_result_docs])

            tt_pseudo_doc = {term: sum([exp_score * scorer.score(term, exp_doc) for exp_doc, exp_score in
                                        normalize_results_scores(tt_results)]) for term in tt_vocab}
            pseudo_pseudo_doc = {term: sum([exp_score * scorer.score(term, exp_doc) for exp_doc, exp_score in
                                            normalize_results_scores(pseudo_results)]) for term in pseudo_vocab}

            tt_qrels = Qrels()
            tt_qrels._qrels[docno] = collections.Counter(tts)

            results_jaccard = jaccard_similarity(tt_result_docnos, pseudo_result_docnos)
            pseudo_results_recall = recall(pseudo_result_docnos, tt_result_docnos)
            cosine = cosine_similarity(tt_pseudo_doc, pseudo_pseudo_doc)
            pseudo_ap = average_precision(docno, sorted(pseudo_query.vector.keys(), key=lambda k:
                                                        pseudo_query.vector[k], reverse=True), tt_qrels)
            pseudo_term_recall = recall(set(pseudo_query.vector.keys()), set(tts))

            print(user, docno, results_jaccard, pseudo_results_recall, cosine, pseudo_ap, pseudo_term_recall, sep=',')
            # print(docno, results_jaccard, pseudo_results_recall, cosine, pseudo_ap, pseudo_term_recall, sep=',')


if __name__ == '__main__':
    main()
