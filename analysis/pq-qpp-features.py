import argparse
import collections
import math
import statistics

import pyndri

from retrieval.core import IndexWrapper, build_rm1, Stopper, Query
from retrieval.scoring import clarity, DirichletTermScorer

"""
List of features:
- clarity
- weighted information gain
- normalized query commitment
"""


def main():
    options = argparse.ArgumentParser()
    options.add_argument('pseudo_queries')
    options.add_argument('expansion_index')
    options.add_argument('stoplist')
    args = options.parse_args()

    pseudo_queries = collections.defaultdict(collections.Counter)
    with open(args.pseudo_queries) as f:
        for line in f:
            docno, term, weight = line.strip().split(',')
            pseudo_queries[docno][term] = float(weight)

    stopper = Stopper(file=args.stoplist)

    index = IndexWrapper(pyndri.Index(args.expansion_index))
    for docno in pseudo_queries:
        query = Query(docno, vector=pseudo_queries[docno])

        top_results = index.query(query, count=10)

        rm1 = build_rm1(top_results, index, stopper=stopper)

        # Features
        rm1_clarity = clarity(rm1.vector, index)
        weighted_ig = wig(query, index, top_results=top_results)
        normalized_qc = nqc(query, index, top_results=top_results)
        average_idf = avg_idf(query.vector.keys(), index)
        simple_clarity = scs(query, index)
        average_scq = statistics.mean(scqs(query, index))

        print(query.title, rm1_clarity, weighted_ig, normalized_qc, average_idf, simple_clarity, average_scq, sep=',')


def scqs(query, index):
    def scq(term):
        try:
            return (1 + math.log(index.term_count(term) + 1)) * math.log(index.total_docs() / (
                    index.term_document_frequency(term) + 1))
        except KeyError:
            return (1 + math.log(index.term_count(term) + 1)) * math.log(index.total_docs() / 1)
    return [scq(term) for term in query.vector.keys()]


def scs(query, index):
    def q_prob(t):
        return query.vector[t] / query.length()

    scs = 0.0
    for term in query.vector.keys():
        try:
            scs += q_prob(term) * math.log(q_prob(term) / (index.term_count(term) / index.total_terms()))
        except ZeroDivisionError:
            scs += 0.0
    return scs


def avg_idf(terms, index):
    total = 0.0
    for term in terms:
        try:
            total += math.log(index.total_docs() / index.term_document_frequency(term))
        except:
            total += 0.0
    return total / len(terms)


def wig(query, index, top_results=None):
    if top_results is None:
        top_results = index.query(query, count=10)
    if len(top_results) == 0:
        return 0.0

    dirichlet_scorer = DirichletTermScorer(index)

    wig = 0.0
    for doc, _ in top_results:
        for term in query.vector:
            p_doc = dirichlet_scorer.score(term, doc)
            p_col = (index.term_count(term) + 1) / index.total_terms()
            lam = 1 / math.sqrt(query.length())
            wig += lam * math.log(p_doc / p_col)

    return wig / len(top_results)


def nqc(query, index, top_results=None):
    if top_results is None:
        top_results = index.query(query, count=10)
    if len(top_results) == 0:
        return 0.0

    mu = statistics.mean([score for _, score in top_results])
    squared_diffs = [(score - mu) ** 2 for _, score in top_results]
    num = math.sqrt(sum(squared_diffs) / len(top_results))

    col_score = 0.0
    for term in query.vector:
        q_weight = query.vector[term] / query.length()
        term_score = math.log((index.term_count(term) / index.total_terms()) + 0.001)
        col_score += q_weight * term_score

    return num / col_score


if __name__ == '__main__':
    main()
