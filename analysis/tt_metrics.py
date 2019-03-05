import argparse
from functools import lru_cache

import pyndri

from retrieval.core import IndexWrapper, ExpandableDocument, Stopper
from retrieval.scoring import DirichletTermScorer, ExpansionDocTermScorer
from tt_pq_metrics import combine_vectors


@lru_cache(maxsize=2**9)
def as_document(docno, target_index, expansion_index):
    return ExpandableDocument(docno, target_index, expansion_index=expansion_index)


def main():
    options = argparse.ArgumentParser()
    options.add_argument('topic_terms')
    options.add_argument('target_index')
    options.add_argument('expansion_index')
    options.add_argument('stoplist')
    args = options.parse_args()

    target_index = IndexWrapper(pyndri.Index(args.target_index))
    expansion_index = IndexWrapper(pyndri.Index(args.expansion_index))

    stopper = Stopper(file=args.stoplist)

    dirichlet_scorer = DirichletTermScorer(target_index)
    expansion_scorer = ExpansionDocTermScorer(DirichletTermScorer(expansion_index))

    with open(args.topic_terms) as f:
        for line in f:
            user, docno, _, term = line.strip().split(',')
            doc = as_document(docno, target_index, expansion_index)

            expansion_docs = doc.expansion_docs(doc.pseudo_query(stopper=stopper))
            expansion_pseudo_doc = combine_vectors(*[r[0].document_vector() for r in expansion_docs])

            orig_score = dirichlet_scorer.score(term, doc)
            expansion_score = expansion_scorer.score(term, expansion_docs)
            appears_in_doc = doc.document_vector()[term] / sum(list(doc.document_vector().values()))
            appears_in_expansion_docs = expansion_pseudo_doc[term] / sum(list(expansion_pseudo_doc.values()))

            print(user, docno, term, orig_score, expansion_score, appears_in_doc, appears_in_expansion_docs, sep=',')


if __name__ == '__main__':
    main()
