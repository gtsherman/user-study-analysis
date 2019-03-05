import argparse
import collections
from pprint import pprint

import pyndri

from retrieval.core import IndexWrapper, read_queries, Qrels, Query
from retrieval.scoring import jaccard_similarity, recall


def main():
    options = argparse.ArgumentParser()
    options.add_argument('topic_terms')
    options.add_argument('queries')
    options.add_argument('qrels')
    options.add_argument('index')
    args = options.parse_args()

    index = IndexWrapper(pyndri.Index(args.index))

    topic_terms = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(args.topic_terms) as f:
        for line in f:
            user, docno, _, term = line.strip().split(',')
            topic_terms[docno][user].append(term)

    queries = read_queries(args.queries)
    qrels = Qrels(file=args.qrels)

    for query in queries:
        judged_docs = qrels.judged_docs(query.title)
        judged_with_tt = judged_docs & set(topic_terms.keys())

        if judged_with_tt:
            query_results = index.query(query, count=10)
            query_results_docs = [r[0].docno for r in query_results]

        for docno in judged_with_tt:
            for user in topic_terms[docno]:
                tt_set = set(topic_terms[docno][user])
                qt_set = set(query.vector.keys())

                tt_query = Query(docno, vector=collections.Counter(topic_terms[docno][user]))
                tt_results = index.query(tt_query, count=10)
                tt_results_docs = [r[0].docno for r in tt_results]

                tt_query_jaccard = jaccard_similarity(tt_set, qt_set)
                tt_query_recall = recall(tt_set, qt_set)
                results_jaccard = jaccard_similarity(set(tt_results_docs), set(query_results_docs))
                results_recall = recall(set(tt_results_docs), set(query_results_docs))

                print(user, docno, query.title, qrels.relevance_of(docno, query.title), tt_query_jaccard,
                      tt_query_recall, results_jaccard, results_recall, sep=',')


if __name__ == '__main__':
    main()
