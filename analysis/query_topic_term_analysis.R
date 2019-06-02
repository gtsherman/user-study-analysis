library(tidyverse)

# Among relevant documents, take the query terms that are not offered as topic term choices.
# How often do these terms appear in the document that they (could not be selected to) describe?
query_term_choice_stemmed %>% 
  inner_join(qrels %>% 
               rename(doc = docno)) %>% 
  filter(rel > 0) %>% 
  inner_join(topics_stopped_stemmed) %>% 
  anti_join(term_choices_stemmed) %>% 
  select(doc, query, term) %>% 
  left_join(doc_vecs_stemmed) %>% 
  mutate(freq = replace_na(freq, 0)) %>%
  arrange(query, doc, term) %>%
  ggplot(aes(freq)) +
  geom_histogram(binwidth = 1) +
  theme_bw() +
  theme(text = element_text(size = 20)) +
  labs(x = 'Term count in document',
       y = 'Freq.') +
  scale_x_continuous(breaks = seq(0, 5))

query_term_choice_stemmed %>%
  inner_join(qrels %>%
               rename(doc = docno)) %>%
  filter(rel > 0) %>%
  inner_join(topics_stopped_stemmed) %>%
  inner_join(term_choices_stemmed) %>%
  distinct() %>% # because stemmed term choices sometimes include multiples of stems, e.g. campaign and campaigns both -> campaign
  select(doc, query, term) %>%
  left_join(doc_vecs_stemmed %>% 
              group_by(doc) %>%
              mutate(prob = freq / sum(freq))) %>%
  mutate(freq = replace_na(freq, 0),
         prob = replace_na(prob, 0)) %>%
  ggplot(aes(freq)) +
  geom_histogram()
