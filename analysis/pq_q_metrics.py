import argparse
import collections
import sys

import pyndri

from retrieval.core import Qrels, Query, IndexWrapper, Stopper
from retrieval.scoring import recall, average_precision, jaccard_similarity, build_vocab, DirichletTermScorer, \
    cosine_similarity, precision


def main():
    options = argparse.ArgumentParser()
    options.add_argument('pseudo_queries')
    options.add_argument('queries')
    options.add_argument('qrels')
    options.add_argument('stoplist')
    options.add_argument('--index')
    args = options.parse_args()

    if args.index:
        index = IndexWrapper(pyndri.Index(args.index))
        scorer = DirichletTermScorer(index)

    qrels = Qrels(file=args.qrels)

    stopper = Stopper(file=args.stoplist)

    judged = collections.defaultdict(set)
    with open(args.qrels) as f:
        for line in f:
            query, _, doc, _ = line.split()
            judged[doc].add(query)

    pq = collections.defaultdict(dict)
    with open(args.pseudo_queries) as f:
        for line in f:
            doc, term, weight = line.strip().split(',')
            pq[doc][term] = float(weight)

    q = collections.defaultdict(set)
    with open(args.queries) as f:
        for line in f:
            query, term = line.strip().split(',')
            q[query].add(term)

    def normalize_results_scores(results):
        total = sum([score for _, score in results])
        return [(doc, score / total) for doc, score in results]

    col_names = 'doc,query,pq_q_recall,pq_q_ap,q_weight_perc'
    if args.index:
        col_names += ',pq_q_results_jacc,pq_q_results_cosine,pq_results_ap,q_results_ap,pq_results_prec,q_results_prec'
    print(col_names)
    for doc in pq:
        pq_query = Query(doc, vector=collections.Counter(pq[doc]))
        if args.index:
            pq_results = index.query(pq_query, 10)
            pq_results_set = set([r.docno for r, _ in pq_results])
        for associated_query in judged[doc]:
            q_query = Query(associated_query, vector=stopper.stop(collections.Counter(q[associated_query])))
            if args.index:
                q_results = index.query(q_query, 10)
                q_results_set = set([r.docno for r, _ in q_results])
                results_jacc = jaccard_similarity(pq_results_set, q_results_set)

                pq_results_ap = average_precision(associated_query, [r.docno for r, _ in pq_results], qrels)
                q_results_ap = average_precision(associated_query, [r.docno for r, _ in q_results], qrels)
                pq_results_prec = precision(pq_results_set, qrels.rel_docs(associated_query))
                q_results_prec = precision(q_results_set, qrels.rel_docs(associated_query))

                pq_vocab = build_vocab(*[r.document_vector() for r, _ in pq_results])
                q_vocab = build_vocab(*[r.document_vector() for r, _ in q_results])

                pq_pseudo_doc = {term: sum([exp_score * scorer.score(term, exp_doc) for exp_doc, exp_score in
                                            normalize_results_scores(pq_results)]) for term in pq_vocab}
                q_pseudo_doc = {term: sum([exp_score * scorer.score(term, exp_doc) for exp_doc, exp_score in
                                           normalize_results_scores(q_results)]) for term in q_vocab}
                cosine = cosine_similarity(pq_pseudo_doc, q_pseudo_doc)

            q_qrels = Qrels()
            q_qrels._qrels[associated_query] = q_query.vector

            pseudo_ap = average_precision(associated_query, sorted(pq[doc].keys(), key=lambda k: pq[doc][k],
                                                                   reverse=True), q_qrels)
            pq_q_recall = recall(set(pq[doc].keys()), q[associated_query])
            q_weight_perc = sum([pq[doc][term] if term in pq[doc] else 0.0 for term in q[associated_query]]) / \
                            sum([pq[doc][term] for term in pq[doc]])

            output = [doc, associated_query, str(pq_q_recall), str(pseudo_ap), str(q_weight_perc)]
            if args.index:
                output += [str(results_jacc), str(cosine), str(pq_results_ap), str(q_results_ap), str(pq_results_prec),
                           str(q_results_prec)]
            print(','.join(output))


if __name__ == '__main__':
    main()
