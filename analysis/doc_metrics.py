import argparse
import collections
import math
from pprint import pprint

import pyndri

from retrieval.core import IndexWrapper, Stopper, Qrels, ExpandableDocument, read_queries, Query
from retrieval.scoring import DirichletTermScorer, QLQueryScorer, ExpansionDocTermScorer, InterpolatedTermScorer, \
    build_vocab, cosine_similarity


def entropy(vector):
    total = sum(vector.values())
    return -1 * sum([vector[k] / total * math.log2(vector[k] / total) for k in vector])


def main():
    options = argparse.ArgumentParser()
    options.add_argument('topic_terms')
    options.add_argument('document')
    options.add_argument('target_index')
    options.add_argument('expansion_index')
    options.add_argument('queries')
    options.add_argument('qrels')
    options.add_argument('stoplist')
    options.add_argument('optimal_params')
    args = options.parse_args()

    optimal_params = collections.defaultdict(dict)
    with open(args.optimal_params) as f:
        for line in f:
            query, param_blob = line.strip().split()
            params = param_blob.split(',')
            for param in params:
                name, value = param.split(':')
                if name == 'origW':
                    optimal_params[query]['o'] = float(value)
                elif name == 'expDocs':
                    optimal_params[query]['d'] = int(value)
                elif name == 'expTerms':
                    optimal_params[query]['t'] = int(value)

    target_index = IndexWrapper(pyndri.Index(args.target_index))
    expansion_index = IndexWrapper(pyndri.Index(args.expansion_index))
    queries = {q.title: q for q in read_queries(args.queries)}
    qrels = Qrels(file=args.qrels)
    stopper = Stopper(file=args.stoplist)

    target_term_scorer = DirichletTermScorer(target_index)
    target_ql_scorer = QLQueryScorer(target_term_scorer)

    topic_terms = collections.defaultdict(lambda: collections.defaultdict(set))
    with open(args.topic_terms) as f:
        for line in f:
            user, docno, _, term = line.strip().split(',')
            topic_terms[docno][user].add(term)

    #print('docno,idtype,idvalue,metric,value')

    docno = args.document

    doc = ExpandableDocument(docno, target_index, expansion_index=expansion_index)
    expansion_docs = doc.expansion_docs(doc.pseudo_query(stopper=stopper))

    pairwise_cosine = []
    for i in range(len(expansion_docs)):
        for j in range(i+1, len(expansion_docs)):
            pairwise_cosine.append(cosine_similarity(expansion_docs[i][0].document_vector(),
                                                 expansion_docs[j][0].document_vector()))
    pairwise_cosine = sum(pairwise_cosine) / len(pairwise_cosine)

    print(docno, 'NA', 'NA', 'pairwise_cosine', pairwise_cosine)

    associated_queries = qrels.judged_for_queries(docno).keys()
    for associated_query in associated_queries:
        query = queries[associated_query]
        query.vector = stopper.stop(query.vector)

        expansion_term_scorer = ExpansionDocTermScorer(DirichletTermScorer(expansion_index), stopper=stopper,
                                                       num_docs=optimal_params[query.title]['d'],
                                                       num_terms=optimal_params[query.title]['t'])
        expansion_ql_scorer = QLQueryScorer(expansion_term_scorer)
        interpolated_term_scorer = InterpolatedTermScorer([target_term_scorer, expansion_term_scorer],
                                                          [optimal_params[query.title]['o'],
                                                           1.0-optimal_params[query.title]['o']])
        interpolated_ql_scorer = QLQueryScorer(interpolated_term_scorer)

        target_ql = target_ql_scorer.score(query, doc)
        expansion_ql = expansion_ql_scorer.score(query, doc)
        expanded_ql = interpolated_ql_scorer.score(query, doc)

        print(docno, 'query', query.title, 'target_ql', target_ql)
        print(docno, 'query', query.title, 'expansion_ql', expansion_ql)
        print(docno, 'query', query.title, 'expanded_ql', expanded_ql)

    expansion_vocab = build_vocab(*[d.document_vector() for d, _ in expansion_docs])
    expansion_lm = {}
    for term in expansion_vocab:
        expansion_lm[term] = expansion_term_scorer.score(term, doc)

    target_lm = {}
    for term in doc.document_vector().keys():
        target_lm[term] = target_term_scorer.score(term, doc)

    distance = cosine_similarity(target_lm, expansion_lm)

    print(docno, 'NA', 'NA', 'distance', distance)

    expansion_entropy = entropy(expansion_lm)
    target_entropy = entropy(target_lm)

    print(docno, 'NA', 'NA', 'expansion_entropy', expansion_entropy)
    print(docno, 'NA', 'NA', 'target_entropy', target_entropy)

    for user in topic_terms[docno]:
        tt = topic_terms[docno][user]
        tt_query = Query(user, vector={term: 1 for term in tt})

        target_tt_likelihood = target_ql_scorer.score(tt_query, doc)
        expansion_tt_likelihood = expansion_ql_scorer.score(tt_query, doc)
        expanded_tt_likelihood = interpolated_ql_scorer.score(tt_query, doc)

        print(docno, 'user', user, 'target_tt_likelihood', target_tt_likelihood)
        print(docno, 'user', user, 'expansion_tt_likelihood', expansion_tt_likelihood)
        print(docno, 'user', user, 'expanded_tt_likelihood', expanded_tt_likelihood)


if __name__ == '__main__':
    main()
