import argparse
import collections


def main():
    options = argparse.ArgumentParser()
    options.add_argument('term_choices')
    options.add_argument('query_terms')
    options.add_argument('annotated_docs')
    options.add_argument('stoplist')
    args = options.parse_args()

    stopwords = set()
    with open(args.stoplist) as f:
        for line in f:
            stopwords.add(line.strip())

    annotated = collections.defaultdict(set)
    with open(args.annotated_docs) as f:
        for line in f:
            doc, query, _ = line.split(',')
            annotated[doc].add(query)

    term_choices = collections.defaultdict(set)
    with open(args.term_choices) as f:
        for line in f:
            doc, term = line.strip().split(',')
            if doc in annotated:
                term_choices[doc].add(term)

    query_terms = collections.defaultdict(set)
    with open(args.query_terms) as f:
        for line in f:
            query, term = line.strip().split(',')
            if term not in stopwords:
                query_terms[query].add(term)

    print('doc,query,num_q_in_choices,perc_q_in_choices')
    for doc in annotated:
        choices = term_choices[doc]
        for query in annotated[doc]:
            q_terms = query_terms[query]

            q_terms_in_choices = choices & q_terms
            print(','.join([doc, query, str(len(q_terms_in_choices)), str(len(q_terms)), str(len(q_terms_in_choices) /
                                                                                             len(q_terms))]))


if __name__ == '__main__':
    main()
