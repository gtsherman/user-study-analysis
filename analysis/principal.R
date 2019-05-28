# Do topic terms get better?
doc_tt_metrics %>%
  ggplot(aes(expanded_tt_likelihood - target_tt_likelihood)) +
    geom_histogram()

## They get better in the expanded LM, but worse in the expansion LM
t.test(doc_tt_metrics$expanded_tt_likelihood, doc_tt_metrics$target_tt_likelihood, paired=T) # better
t.test(doc_tt_metrics$expansion_tt_likelihood, doc_tt_metrics$target_tt_likelihood, paired=T) # worse

## Topic term change is approximately equal across users
doc_tt_metrics %>%
  ggplot(aes(user, expanded_tt_likelihood - target_tt_likelihood, fill = user)) +
  geom_boxplot()

## Topic term relative inclusion in expansion docs
tt_metrics %>% 
  mutate(in_doc = sign(percent_of_doc), 
         in_exp = sign(percent_of_exp), 
         added = in_exp - in_doc) %>% 
  group_by(user, docno) %>% 
  summarize(perc_added = sum(added) / n()) %>% 
  ggplot(aes(perc_added)) + 
  geom_histogram(binwidth = .1) +
  theme_bw() +
  theme(text = element_text(size = 20)) + 
  labs(x = 'Relative percent of topic terms in expansion docs',
       y = 'Freq.')

## There's no particular relationship between term overlap and topic term likelihood change
doc_tt_metrics %>% 
  mutate(likelihood_diff = expanded_tt_likelihood - target_tt_likelihood) %>%
  inner_join(tt_pq_metrics) %>%
  summarize(cor(pseudo_ap, likelihood_diff, method='kendall'),
            cor(pseudo_term_recall, likelihood_diff, method='kendall'))

## Very slight relationship between results overlap and topic term likelihood change.
## Meaning the topic terms increase in probability more when the topic terms and the pseudo-query
## retrieve the same documents.
## Indicates it's more important that the pseudo-query retrieve the same documents as the topic terms
## than that the pseudo-query resemble the topic terms.
doc_tt_metrics %>% 
  mutate(likelihood_diff = expanded_tt_likelihood - target_tt_likelihood) %>%
  inner_join(tt_pq_metrics) %>%
  summarize(cor(results_jacc, likelihood_diff),
            cor(pseudo_results_recall, likelihood_diff, method='kendall'),
            cor(cosine, likelihood_diff, method='kendall'))


# Do query terms get better?
## They are always better for relevant docs, regardless of LM
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(rel = sign(rel)) %>%
  gather(ql, val, expanded_ql:target_ql) %>%
  ggplot(aes(factor(rel), val)) +
    facet_wrap(~ ql) +
    geom_boxplot()

## They almost always improve for both relevant and nonrelevant docs, 
## but they seem to improve more for relevant docs more than nonrels.
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(rel = sign(rel)) %>%
  ggplot(aes(factor(rel), expanded_ql - target_ql)) +
    geom_boxplot()

## Confirmed. They improve more for relevant docs than for nonrels.
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(rel = sign(rel)) %>%
  with(t.test(expanded_ql - target_ql ~ factor(rel))) # they 

## But there's still no particular association between term overlap and query likelihood improvement.
## Though I guess it's twice as strong for rel docs, but still not anything to speak of.
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(ql_diff = expanded_ql - target_ql,
         ql_diff2 = expansion_ql - target_ql,
         rel = sign(rel)) %>%
  inner_join(tt_pq_metrics) %>%
  group_by(rel) %>%
  summarize(cor(pseudo_ap, ql_diff, method='kendall'),
            cor(pseudo_term_recall, ql_diff, method='kendall'),
            cor(pseudo_ap, expanded_ql, method='kendall'),
            cor(pseudo_term_recall, expanded_ql, method='kendall'),
            cor(pseudo_ap, target_ql, method='kendall'),
            cor(pseudo_term_recall, target_ql, method='kendall'))

## Again, there's a very slight correlation between results overlap and the extent of QL improvement.
## It's not very strong, but it indicates again that the most important thing is retrieving the same documents
## rather than term overlap.
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(ql_diff = expanded_ql - target_ql,
         ql_diff2 = expansion_ql - target_ql,
         rel = sign(rel)) %>%
  inner_join(tt_pq_metrics) %>%
  group_by(rel) %>%
  summarize(cor(results_jacc, ql_diff, method='kendall'),
            cor(pseudo_results_recall, ql_diff, method='kendall'),
            cor(cosine, ql_diff, method='kendall'),
            cor(results_jacc, ql_diff2, method='kendall'),
            cor(pseudo_results_recall, ql_diff2, method='kendall'),
            cor(cosine, ql_diff2, method='kendall'),
            cor(results_jacc, pseudo_results_recall, method='kendall'),
            cor(results_jacc, cosine, method='kendall'),
            cor(pseudo_results_recall, cosine, method='kendall'))


# Are expansion documents more coherent when topic terms and pseudo-query are similar?
## Same as before, they're not more coherent according to term overlap, but they are a bit more coherent
## when the topic terms and the pseudo-query results overlap.
doc_q_metrics %>%
  inner_join(doc_only_metrics) %>%
  inner_join(tt_pq_metrics) %>%
  summarize(cor(pseudo_term_recall, pairwise_cosine, method='kendall'),
            cor(pseudo_ap, pairwise_cosine, method='kendall'),
            cor(results_jacc, pairwise_cosine, method='kendall'),
            cor(pseudo_results_recall, pairwise_cosine, method='kendall'))


# Metrics of term overlap correlate better with metrics of result overlap when they account for the
# prominence of the topic terms in the pseudo-query (i.e. pseudo_ap correlates better than pseudo_term_recall).
# Cosine b/w the tt pseudo-doc and pq pseudo-doc correlates even better than pseudo_ap, which makes sense b/c
# it is a way of measuring the similarity of the results lists without requiring certain documents, i.e.
# different documents are fine as long as they have similar makeup.
tt_pq_metrics %>%
  select(-user, -docno) %>%
  correlate(method = 'kendall')

tt_pq_metrics %>%
  mutate(diff = pseudo_term_recall - pseudo_results_recall) %>%
  arrange(-diff)
