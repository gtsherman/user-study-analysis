# Topic term likelihood change histogram
doc_tt_metrics %>%
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  ggplot(aes(expanded_tt_likelihood - target_tt_likelihood)) +
  geom_histogram(bins = 20) +
  geom_vline(xintercept = 0, color = 'red') +
  labs(x = 'Change in likelihood', y = 'Freq.') +
  theme_bw() +
  theme(text = element_text(size=20))

# Topic term likelihood change t-test
doc_tt_metrics %>%
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  with(t.test(expanded_tt_likelihood, target_tt_likelihood, paired = T))

# Topic term likelihiood change per annotator boxplot
doc_tt_metrics %>%
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  ggplot(aes(user, expanded_tt_likelihood - target_tt_likelihood, fill = user)) +
  geom_boxplot() +
  scale_x_discrete(labels = c('A', 'B', 'C', 'D', 'E', 'F', 'G')) +
  theme_bw() +
  scale_fill_brewer(palette = 'BrBG', type = 'qual', guide = F) +
  labs(x = 'Annotator', y = 'Change in likelihood') +
  theme(text = element_text(size=20))

# Topic term likelihood change per annotator ANOVA
doc_tt_metrics %>%
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  with(summary(aov(expanded_tt_likelihood - target_tt_likelihood ~ user)))



# Query likelihood change per relevance per LM boxplot
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  #gather(m, v, expanded_ql:target_ql) %>% 
  #mutate(v = exp(v)) %>%
  #spread(m, v) %>%
  mutate(rel = sign(rel)) %>%  # convert any level of relevance into 1
  gather(ql, val, expanded_ql, target_ql) %>%
  mutate(ql = ifelse(ql == 'expanded_ql', 'Expanded Language Model', 'Target Language Model')) %>%
  mutate(ql = factor(ql, levels = c('Target Language Model', 'Expanded Language Model'))) %>%
  ggplot(aes(factor(rel), val)) +
  facet_wrap(~ ql) +#, labeller = as_labeller(c('expanded_ql' = 'Expanded Language Model', 'target_ql' = 'Target Language Model'))) +
  geom_boxplot() +
  scale_x_discrete(labels = c('Nonrel.', 'Rel.')) +
  theme_bw() +
  labs(x = 'Relevance', y = 'Query log likelihood') +
  theme(text = element_text(size=20))

# Query likelihood change per relevance boxplot
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(rel = sign(rel)) %>%
  mutate(ql_diff = expanded_ql - target_ql) %>%
  ggplot(aes(factor(rel), ql_diff)) +
  geom_boxplot() +
  scale_x_discrete(labels = c('Nonrel.', 'Rel.')) +
  theme_bw() +
  labs(x = 'Relevance', y = 'Query log likelihood difference') +
  theme(text = element_text(size=20))


# Expansion coherence 
# ...


# Topic term/pseudo-query overlap
# TT/PQ recall vs. TT likelihood change plot
doc_tt_metrics %>% 
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  mutate(likelihood_diff = expanded_tt_likelihood - target_tt_likelihood) %>%
  inner_join(tt_pq_metrics) %>%
  ggplot(aes(pseudo_term_recall, likelihood_diff)) +
  geom_point() +
  theme_bw() +
  labs(x = 'Recall', y = 'Topic term likelihood change') +
  geom_hline(yintercept = 0.0, color = 'red') +
  theme(text = element_text(size=20))

# TT/PQ results jaccard vs. TTL
doc_tt_metrics %>% 
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  mutate(likelihood_diff = expanded_tt_likelihood - target_tt_likelihood) %>%
  inner_join(tt_pq_metrics) %>%
  with(cor.test(results_jacc, likelihood_diff, method='kendall'))

# TT/PQ results similarity vs. TTL
doc_tt_metrics %>% 
  gather(m, v, expanded_tt_likelihood:target_tt_likelihood) %>%
  mutate(v = exp(v)) %>%
  spread(m, v) %>%
  mutate(likelihood_diff = expanded_tt_likelihood - target_tt_likelihood) %>%
  inner_join(tt_pq_metrics) %>%
  with(cor.test(cosine, likelihood_diff, method='kendall'))

# TT/PQ recall vs. QL change
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(expanded_ql = exp(expanded_ql),
         expansion_ql = exp(expansion_ql),
         target_ql = exp(target_ql),
         ql_diff = expanded_ql - target_ql,
         ql_diff_expansion = expansion_ql - target_ql,
         rel = sign(rel)) %>%
  inner_join(tt_pq_metrics) %>%
  filter(rel > 0) %>%
  #group_by(rel) %>%
  with(cor.test(pseudo_ap, ql_diff, method='kendall'))

# TT/PQ results overlap vs. QL change
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(expanded_ql = exp(expanded_ql),
         expansion_ql = exp(expansion_ql),
         target_ql = exp(target_ql),
         ql_diff = expanded_ql - target_ql,
         ql_diff_expansion = expansion_ql - target_ql,
         rel = sign(rel)) %>%
  inner_join(tt_pq_metrics) %>%
  filter(rel > 0) %>%
  with(cor.test(results_jacc, ql_diff, method='kendall'))

# TT/PQ results similarity vs. QL change
doc_q_metrics %>%
  inner_join(tt_q_metrics %>% select(docno, rel) %>% distinct()) %>%
  mutate(expanded_ql = exp(expanded_ql),
         expansion_ql = exp(expansion_ql),
         target_ql = exp(target_ql),
         ql_diff = expanded_ql - target_ql,
         ql_diff_expansion = expansion_ql - target_ql,
         rel = sign(rel)) %>%
  inner_join(tt_pq_metrics) %>%
  filter(rel > 0) %>%
  with(cor.test(cosine, ql_diff, method='kendall'))

# TT/PQ results overlap vs. expansion doc coherence
doc_q_metrics %>%
  inner_join(doc_only_metrics) %>%
  inner_join(tt_pq_metrics) %>%
  with(cor.test(pseudo_results_recall, pairwise_cosine, method='kendall'))

# TT/PQ results similarity vs. expansion doc coherence
doc_q_metrics %>%
  inner_join(doc_only_metrics) %>%
  inner_join(tt_pq_metrics) %>%
  with(cor.test(cosine, pairwise_cosine, method='kendall'))

# Linear models
tt_pq_metrics %>%
  inner_join(doc_tt_metrics) %>% 
  with(summary(lm(tt_change ~ pseudo_term_recall + results_jacc + cosine)))

tt_pq_metrics %>%
  inner_join(doc_only_metrics) %>%
  with(summary(lm(pairwise_cosine ~ pseudo_term_recall + results_jacc + cosine)))
