# Average number of selections per document per user
num_user_selections = tt %>%
  group_by(user, doc) %>%
  tally()
num_user_selections %>%
  summarize(m = mean(n))

# Average number of selections per doc/user box plot
num_user_selections %>%
  ggplot(aes(x = user, y = n)) +
    geom_boxplot()

# 1-Way ANOVA of num selections per doc/user
select_anova = aov(n ~ user, data = num_user_selections)
summary(select_anova)
TukeyHSD(select_anova)

# Total docs per user
tt %>%
  select(user, doc) %>%
  distinct() %>%
  group_by(user) %>%
  tally()

# Total selections per user
num_user_selections %>%
  group_by(user) %>%
  summarize(total = sum(n))

# Cleanup
rm(num_user_selections)
rm(select_anova)