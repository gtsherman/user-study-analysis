library(tidyverse)

# How many annotations of relevant documents contain no query terms?
tt_q_metrics_stemmed %>% 
  filter(rel > 0, 
         tt_q_recall == 0) %>%
  nrow() / 
  tt_q_metrics_stemmed %>%
  filter(rel > 0) %>% 
  nrow()

# How many document annotations that do not include query term are the result of those terms not being offered as choices?
tt_q_metrics_stemmed %>%
  filter(rel > 0,
         tt_q_recall == 0) %>%
  inner_join(query_term_choice_stemmed) %>%
  filter(num_q_in_choices == 0) %>%
  nrow() /
  tt_q_metrics_stemmed %>%
  filter(rel > 0,
         tt_q_recall == 0) %>%
  inner_join(query_term_choice_stemmed) %>%
  nrow()

# Among relevant documents, how many offered no query term choices?
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>%
  filter(rel > 0,
         num_q_in_choices == 0) %>% 
  nrow() /
  query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(rel > 0) %>%
  nrow()

# When documents offer no query terms as topic term choices, how many query terms were available
# to be sampled in the first place?
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(num_q_in_choices == 0, rel > 0) %>% 
  ggplot(aes(q_length)) + 
  geom_histogram(binwidth = 1) + 
  scale_x_continuous(breaks = seq(0, 9))

# Among relevant documents, take the query terms that are not offered as topic term choices.
# How often do these terms appear in the document that they (could not be selected to) describe?
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(rel > 0) %>% 
  inner_join(topics_stopped_stemmed) %>% 
  anti_join(term_choices_stemmed) %>% 
  select(doc, query, term) %>% 
  left_join(doc_vecs_stopped_stemmed) %>% 
  mutate(freq = replace_na(freq, 0)) %>%
  ggplot(aes(freq)) +
  geom_histogram(binwidth = 1) +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  labs(x = 'Term count in document',
       y = 'Freq.') +
  scale_x_continuous(breaks = seq(0, 7))

# The same as above, but limited to only those documents that offered none of the query terms
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(rel > 0,
         perc_q_in_choices == 0) %>% 
  inner_join(topics_stopped_stemmed) %>% 
  anti_join(term_choices_stemmed) %>% 
  select(doc, query, term) %>% 
  left_join(doc_vecs_stopped_stemmed) %>% 
  mutate(freq = replace_na(freq, 0)) %>%
  ggplot(aes(freq)) +
  geom_histogram(binwidth = 1) +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  labs(x = 'Term count in document',
       y = 'Freq.') +
  scale_x_continuous(breaks = seq(0, 7))

# Among relevant documents, take the query terms that _are_ offered as topic term choices.
# How often do these terms appear in the document that they might describe?
query_term_choice_stemmed %>%
  inner_join(qrels %>%
               rename(doc = docno)) %>%
  filter(rel > 0) %>%
  inner_join(topics_stopped_stemmed) %>%
  inner_join(term_choices_stemmed) %>%
  distinct() %>% # because stemmed term choices sometimes include multiples of stems, e.g. campaign and campaigns both -> campaign
  select(doc, query, term) %>%
  left_join(doc_vecs_stopped_stemmed %>% 
              group_by(doc) %>%
              mutate(prob = freq / sum(freq))) %>%
  mutate(freq = replace_na(freq, 0),
         prob = replace_na(prob, 0)) %>%
  ggplot(aes(freq)) +
  geom_histogram(binwidth = 1) +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  labs(x = 'Term count in document',
       y = 'Freq.')

# Among relevant documents where at least some query terms were offered but none were selected...
# summarize the properties of these queries.
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(rel > 0, 
         perc_q_in_choices > 0) %>% 
  inner_join(tt_q_metrics_stemmed) %>% 
  filter(tt_q_recall == 0) %>% 
  select(-user) %>% 
  distinct() %>% 
  summarize(mean(q_length),
            min(q_length),
            mean(perc_q_in_choices),
            mean(num_q_in_choices),
            median(num_q_in_choices))

# Same as above, but without the "no query terms selected" criterion.
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(rel > 0, 
         perc_q_in_choices > 0) %>% 
  inner_join(tt_q_metrics_stemmed) %>% 
  select(-user) %>% 
  distinct() %>% 
  summarize(mean(q_length),
            min(q_length),
            mean(perc_q_in_choices),
            mean(num_q_in_choices),
            median(num_q_in_choices))

# Among relevant documents, take the query terms that _are_ offered as topic term choices, but aren't selected by annotators.
# How often do these terms appear in the document that they might describe?
query_term_choice_stemmed %>%
  inner_join(qrels %>%
               rename(doc = docno)) %>%
  filter(rel > 0) %>%
  inner_join(tt_q_metrics_stemmed) %>%
  inner_join(topics_stopped_stemmed) %>%
  inner_join(term_choices_stemmed) %>%
  distinct() %>% # because stemmed term choices sometimes include multiples of stems, e.g. campaign and campaigns both -> campaign
  anti_join(tt_stemmed) %>%
  distinct() %>%
  #select(doc, query, term) %>%
  left_join(doc_vecs_stopped_stemmed %>% 
              group_by(doc) %>%
              mutate(prob = freq / sum(freq))) %>%
  mutate(freq = replace_na(freq, 0),
         prob = replace_na(prob, 0)) %>%
  ggplot(aes(freq)) +
    geom_histogram(binwidth = 1) +
    theme_bw() +
    theme(text = element_text(size = 20)) +
    labs(x = 'Term count in document',
         y = 'Freq.') +
    xlim(-1, 28)

# Compare the above by annotator
query_term_choice_stemmed %>%
  inner_join(qrels %>%
               rename(doc = docno)) %>%
  filter(rel > 0) %>%
  inner_join(tt_q_metrics_stemmed) %>%
  inner_join(topics_stopped_stemmed) %>%
  inner_join(term_choices_stemmed) %>%
  distinct() %>% # because stemmed term choices sometimes include multiples of stems, e.g. campaign and campaigns both -> campaign
  anti_join(tt_stemmed) %>%
  distinct() %>%
  #select(doc, query, term) %>%
  left_join(doc_vecs_stopped_stemmed %>% 
              group_by(doc) %>%
              mutate(prob = freq / sum(freq))) %>%
  mutate(freq = replace_na(freq, 0),
         prob = replace_na(prob, 0),
         user = factor(user, labels = c('A', 'B', 'C',
                                        'D', 'E', 'F', 'G'))) %>% 
  ggplot(aes(user, freq)) + 
  geom_violin() +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  labs(x = 'Annotator',
       y = 'Freq. of selected query terms in doc. text')
# Also compare with documents that *are* selected
query_term_choice_stemmed %>%
  inner_join(qrels %>%
               rename(doc = docno)) %>%
  filter(rel > 0) %>%
  inner_join(tt_q_metrics_stemmed) %>%
  inner_join(topics_stopped_stemmed) %>%
  inner_join(term_choices_stemmed) %>%
  distinct() %>% # because stemmed term choices sometimes include multiples of stems, e.g. campaign and campaigns both -> campaign
  inner_join(tt_stemmed) %>%
  distinct() %>%
  #select(doc, query, term) %>%
  left_join(doc_vecs_stopped_stemmed %>% 
              group_by(doc) %>%
              mutate(prob = freq / sum(freq))) %>%
  mutate(freq = replace_na(freq, 0),
         prob = replace_na(prob, 0),
         user = factor(user, labels = c('A', 'B', 'C',
                                        'D', 'E', 'F', 'G'))) %>% 
  ggplot(aes(user, freq)) + 
  geom_violin() +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  labs(x = 'Annotator',
       y = 'Freq. of selected query terms in doc. text')

# Can we predict relevance from TT/Q overlap?
tt_q_metrics %>% 
  select(-tt_q_recall, -tt_q_jacc) %>% # clear these since the stemmed versions are more informative
  inner_join(tt_q_metrics_stemmed) %>% 
  with(train(factor(rel > 0) ~ tt_q_recall + tt_q_results_recall, 
             data = ., 
             method = 'glm', family = binomial, 
             trControl = trainControl('cv', 10)))