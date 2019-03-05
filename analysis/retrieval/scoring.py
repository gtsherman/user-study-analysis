import math


class DirichletTermScorer(object):
    def __init__(self, index, mu=2500, epsilon=1.0):
        self.index = index
        self.mu = mu
        self.epsilon = epsilon

    def score(self, term, document):
        term = term.lower()

        doc_vector = document.document_vector()
        term_freq = doc_vector[term]
        doc_length = sum(doc_vector.values())
        collection_prob = (self.epsilon + self.index.term_count(term)) / self.index.total_terms()
        return (term_freq + self.mu * collection_prob) / (doc_length + self.mu)


class InterpolatedTermScorer(object):
    def __init__(self, scorers):
        """
        :param scorers: A {scorer: weight} dict.
        """
        self.scorers = scorers

    def score(self, term, document):
        return sum([self.scorers[scorer] * scorer.score(term, document) for scorer in self.scorers])


class ExpansionDocTermScorer(object):
    def __init__(self, scorer):
        """
        :param scorer: Should be a scorer that is associated with the expansion index.
        """
        self._scorer = scorer

    def score(self, term, expansion_docs):
        """
        :param term: The term string.
        :param expansion_docs: A list of (Document, float) tuples, where float is the probability score of the
        associated document.
        :return: The term score.
        """
        return sum([exp_score * self._scorer.score(term, exp_doc) for exp_doc, exp_score in expansion_docs])


class QLQueryScorer(object):
    def __init__(self, term_scorer):
        self.term_scorer = term_scorer

    def score(self, query, document):
        score = 0.0
        for term in query.vector:
            q_weight = query.vector[term] / query.length()
            term_score = math.log(self.term_scorer.score(term, document))
            score += q_weight * term_score
        return score


def build_vocab(*vectors):
    vocab = set()
    for vector in vectors:
        vocab = vocab | vector.keys()
    return vocab


def recall(returned, expected):
    return len(returned & expected) / len(expected)


def precision(returned, expected):
    return len(returned & expected) / len(returned)


def jaccard_similarity(set1, set2):
    try:
        return len(set1 & set2) / len(set1 | set2)
    except ZeroDivisionError:
        return 0.0


def kl_divergence(vector1, vector2):
    vector1_length = sum(vector1.values())
    vector2_length = sum(vector2.values())

    if vector1_length == 0 or vector2_length == 0:
        return 0.0

    total = 0.0
    for term in vector1:  # don't need to include vector2 because if it's not in vector1, P(t) = 0.0 -> KL = 0.0
        p_term = vector1[term] / vector1_length
        q_term = vector2[term] / vector2_length
        if q_term == 0:
            total += 0.0
        else:
            total += p_term * math.log(p_term / q_term)

    return total


def clarity(vector, index):
    vector_length = sum(vector.values())

    if vector_length == 0:
        return 0.0

    total = 0.0
    for term in vector:
        p_term = vector[term] / vector_length
        q_term = (index.term_count(term) + 1) / index.total_terms()
        total += p_term * math.log(p_term / q_term)

    return total


def cosine_similarity(vector1, vector2):
    vector1_length = sum(vector1.values())
    vector2_length = sum(vector2.values())

    num = 0.0
    for term in vector1.keys() & vector2.keys():  # use intersection to avoid terms missing in either, they add nothing
        num += vector1[term] / vector1_length * vector2[term] / vector2_length

    denom1 = math.sqrt(sum([(x / vector1_length)**2 for x in vector1.values()]))
    denom2 = math.sqrt(sum([(x / vector2_length)**2 for x in vector2.values()]))

    try:
        return num / (denom1 * denom2)
    except ZeroDivisionError:
        return 0.0


def average_precision(query_title, search_results, qrels, rank_cutoff=None):
    if rank_cutoff:
        search_results = search_results[:rank_cutoff]

    ap = 0.0
    rels = 0

    for i, search_result in enumerate(search_results):
        if qrels.is_rel(search_result, query_title):
            rels += 1
            ap += rels / float(i+1)

    try:
        ap /= len(qrels.rel_docs(query_title))
    except ZeroDivisionError:
        ap = 0.0

    return ap
