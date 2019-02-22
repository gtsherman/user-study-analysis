irr_tt = tt %>%
  filter(doc %in% irr_docs)

# Average number of terms per user per document
irr_tt %>%
  group_by(user, doc) %>%
  tally() %>%
  summarize(m = mean(n))

# Fleiss' kappa where each term is the subject and the ratings are yes/no (select/ignore).
# This is perhaps an unreasonable use of Fleiss' kappa because users weren't allowed to select
# all terms, but it could still be argued that they actively chose not to select that term in
# preference of another term.
irr_tt %>%
  group_by(doc, term) %>%
  tally() %>%
  ungroup() %>%
  right_join(term_choices %>%
               filter(doc %in% irr_docs)) %>%
  group_by(doc) %>%
  mutate(y = ifelse(is.na(n), 0L, n),
         n = 7 - y,
         P = 1 / (7 * (7 - 1)) * (n * (n - 1) + y * (y - 1))) %>%
  summarize(P = mean(P),
            pn = mean(n) / 7,
            py = mean(y) / 7) %>%
  mutate(pe = pn^2 + py^2,
         kappa = (P - pe) / (1 - pe))
