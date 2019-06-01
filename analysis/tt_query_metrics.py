import argparse
import collections

import pyndri

from retrieval.core import IndexWrapper, read_queries, Qrels, Query, Stopper
from retrieval.scoring import jaccard_similarity, recall


def main():
    options = argparse.ArgumentParser()
    options.add_argument('topic_terms')
    options.add_argument('queries')
    options.add_argument('qrels')
    options.add_argument('index')
    options.add_argument('stoplist')
    options.add_argument('--skip-retrieval', action='store_true')
    args = options.parse_args()

    index = IndexWrapper(pyndri.Index(args.index))
    stopper = Stopper(file=args.stoplist)

    topic_terms = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(args.topic_terms) as f:
        for line in f:
            user, docno, _, term = line.strip().split(',')
            topic_terms[docno][user].append(term)

    queries = read_queries(args.queries, format=args.queries.split('.')[-1])
    qrels = Qrels(file=args.qrels)

    for query in queries:
        judged_docs = qrels.judged_docs(query.title)
        judged_with_tt = judged_docs & set(topic_terms.keys())

        if judged_with_tt and not args.skip_retrieval:
            query_results = index.query(query, count=10)
            query_results_docs = [r[0].docno for r in query_results]

        for docno in judged_with_tt:
            for user in topic_terms[docno]:
                tt_set = set(topic_terms[docno][user]) - stopper.stopwords
                qt_set = set(query.vector.keys()) - stopper.stopwords

                tt_query_jaccard = jaccard_similarity(tt_set, qt_set)
                tt_query_recall = recall(tt_set, qt_set)
                results_jaccard = -1
                results_recall = -1

                if not args.skip_retrieval:
                    tt_query = Query(docno, vector=collections.Counter(topic_terms[docno][user]))
                    tt_results = index.query(tt_query, count=10)
                    tt_results_docs = [r[0].docno for r in tt_results]

                    results_jaccard = jaccard_similarity(set(tt_results_docs), set(query_results_docs))
                    results_recall = recall(set(tt_results_docs), set(query_results_docs))

                print(user, docno, query.title, qrels.relevance_of(docno, query.title), tt_query_jaccard,
                      tt_query_recall, results_jaccard, results_recall, sep=',')


if __name__ == '__main__':
    main()
