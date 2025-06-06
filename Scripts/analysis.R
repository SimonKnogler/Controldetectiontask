# Example R analysis script for the Control Detection Task
# This script loads all CSV files in the Main Experiment/data folder
# and performs signal detection and metacognitive analyses.

# Required packages
library(tidyverse)
library(lme4)
library(ez)
# create directory for plots
if (!dir.exists('Plots')) dir.create('Plots')

# helper: compute d'
dprime_calc <- function(hits, fas, miss, cr) {
  hr <- (hits + 0.5) / (hits + miss + 1)
  far <- (fas + 0.5) / (fas + cr + 1)
  qnorm(hr) - qnorm(far)
}

# directory with participant CSVs
DATA_DIR <- file.path('Main Experiment', 'data')
files <- list.files(DATA_DIR, pattern = '\\.[cC][sS][vV]$', full.names = TRUE)

# load and combine
raw_list <- lapply(files, read.csv)
data <- bind_rows(raw_list)

# keep test trials only
trial_data <- data %>%
  filter(phase == 'test') %>%
  mutate(correct = resp_shape == true_shape,
         expect = factor(expect_level, levels = c('low', 'high')),
         bias = factor(angle_bias))

# compute sensitivity and metacognitive measures per participant and condition
analysis <- trial_data %>%
  group_by(participant, expect, bias) %>%
  group_modify(~{
    df <- .x
    hits <- sum(df$true_shape == 'square' & df$resp_shape == 'square')
    fas  <- sum(df$true_shape == 'dot'    & df$resp_shape == 'square')
    miss <- sum(df$true_shape == 'square' & df$resp_shape == 'dot')
    cr   <- sum(df$true_shape == 'dot'    & df$resp_shape == 'dot')
    dprime <- dprime_calc(hits, fas, miss, cr)
    tibble(
      accuracy = mean(df$correct, na.rm = TRUE),
      dprime = dprime,
      conf = mean(df$conf_level, na.rm = TRUE),
      agency = mean(df$agency_rating, na.rm = TRUE)
    )
  }) %>%
  ungroup()

# repeated measures ANOVAs
anova_conf <- ezANOVA(data = analysis, dv = conf, wid = participant,
                      within = .(expect, bias))
anova_agency <- ezANOVA(data = analysis, dv = agency, wid = participant,
                        within = .(expect, bias))
anova_dprime <- ezANOVA(data = analysis, dv = dprime, wid = participant,
                        within = .(expect, bias))
print(anova_conf)
print(anova_agency)
print(anova_dprime)

# expectation effect during medium control strength (prop_used ~ 0.5)
medium_data <- trial_data %>% filter(abs(prop_used - 0.5) < 1e-6)

expect_effect <- medium_data %>%
  group_by(participant, bias, expect) %>%
  summarise(conf = mean(conf_level, na.rm = TRUE),
            agency = mean(agency_rating, na.rm = TRUE), .groups = 'drop') %>%
  pivot_wider(names_from = expect, values_from = c(conf, agency)) %>%
  mutate(conf_effect = conf_high - conf_low,
         agency_effect = agency_high - agency_low)

baseline <- analysis %>%
  filter(expect == 'low') %>%
  group_by(participant) %>%
  summarise(baseline_dprime = mean(dprime), .groups = 'drop')

effects <- left_join(expect_effect, baseline, by = 'participant')

conf_model <- lmer(conf_effect ~ baseline_dprime * bias + (1|participant), data = effects)
agency_model <- lmer(agency_effect ~ baseline_dprime * bias + (1|participant), data = effects)
print(summary(conf_model))
print(summary(agency_model))

# ----- plotting -----
summary_data <- analysis %>%
  group_by(expect, bias) %>%
  summarise(
    conf_m = mean(conf),
    conf_se = sd(conf)/sqrt(n()),
    agency_m = mean(agency),
    agency_se = sd(agency)/sqrt(n()),
    dprime_m = mean(dprime),
    dprime_se = sd(dprime)/sqrt(n()),
    .groups = 'drop'
  )

conf_plot <- ggplot(summary_data, aes(expect, conf_m, fill = bias)) +
  geom_col(position = position_dodge()) +
  geom_errorbar(aes(ymin = conf_m - conf_se, ymax = conf_m + conf_se),
                width = 0.2, position = position_dodge(width = 0.9)) +
  labs(title = 'Confidence by expectation and bias',
       y = 'Confidence', x = 'Expectation')
ggsave(file.path('Plots', 'conf_plot.png'), conf_plot, width = 6, height = 4)

agency_plot <- ggplot(summary_data, aes(expect, agency_m, fill = bias)) +
  geom_col(position = position_dodge()) +
  geom_errorbar(aes(ymin = agency_m - agency_se, ymax = agency_m + agency_se),
                width = 0.2, position = position_dodge(width = 0.9)) +
  labs(title = 'Agency rating by expectation and bias',
       y = 'Agency rating', x = 'Expectation')
ggsave(file.path('Plots', 'agency_plot.png'), agency_plot, width = 6, height = 4)

dprime_plot <- ggplot(summary_data, aes(expect, dprime_m, fill = bias)) +
  geom_col(position = position_dodge()) +
  geom_errorbar(aes(ymin = dprime_m - dprime_se, ymax = dprime_m + dprime_se),
                width = 0.2, position = position_dodge(width = 0.9)) +
  labs(title = "d' by expectation and bias", y = "d'", x = 'Expectation')
ggsave(file.path('Plots', 'dprime_plot.png'), dprime_plot, width = 6, height = 4)

conf_eff_plot <- ggplot(effects, aes(baseline_dprime, conf_effect, color = bias)) +
  geom_point() +
  geom_smooth(method = 'lm', se = FALSE) +
  labs(title = "Confidence expectation effect vs baseline d'",
       y = 'High - Low confidence', x = "Baseline d'")
ggsave(file.path('Plots', 'conf_effect_plot.png'), conf_eff_plot, width = 6, height = 4)

agency_eff_plot <- ggplot(effects, aes(baseline_dprime, agency_effect, color = bias)) +
  geom_point() +
  geom_smooth(method = 'lm', se = FALSE) +
  labs(title = "Agency expectation effect vs baseline d'",
       y = 'High - Low agency', x = "Baseline d'")
ggsave(file.path('Plots', 'agency_effect_plot.png'), agency_eff_plot, width = 6, height = 4)
