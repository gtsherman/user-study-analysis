# Percent of checks correct per user
quality %>%
  group_by(user) %>%
  summarize(p = sum(selected) / n())

# Percent of checks correct per index
quality_percents = quality %>%
  inner_join(tt %>% 
          select(-term) %>% 
          distinct(),
        by = c('user', 'doc')) %>%
  group_by(user, index) %>%
  summarize(p = sum(selected) / n())
quality_percents %>%
  ggplot(aes(x = index, y = p)) +
    geom_boxplot()

# 1-Way ANOVA, index as factor
qp_anova = aov(p ~ index, data = quality_percents)
summary(qp_anova)
TukeyHSD(qp_anova)

# Percent of checks correct per user
quality_percents %>%
  ggplot(aes(x = user, y = p)) +
  geom_boxplot()

# 1-Way ANOVA, user as factor
qp_anova = aov(p ~ user, data = quality_percents)
summary(qp_anova)

# Cleanup
rm(qp_anova)
rm(quality_percents)