library(readr)

k_hat <- read_csv('/Users/vcerqueira/Dropbox/Research/feature_engineering/fselection/data/FSELECTION_KHAT_LASSO_1.csv')
k_time <- read_csv('/Users/vcerqueira/Dropbox/Research/feature_engineering/fselection/data/FSELECTION_EXECT_LASSO_1.csv')
perf <- read_csv('/Users/vcerqueira/Dropbox/Research/feature_engineering/fselection/data/FSELECTION_ERROR_LASSO_1.csv')

perf <- as.data.frame(perf)
colnames(k_hat) <-
  colnames(perf) <- c(
    "Const-10",
    "Const-2",
    "Const-50",
    "PACF@0.01",
    "PACF@0.005",
    "PACF@0.001",
    "PCA",
    "FNN@0",
    "FNN@0.1",
    "FNN@0.01",
    "GridSearch",
    "AIC",
    "BIC",
    "Freq",
    "FS_Corr",
    "FS_MI",
    #"FS_RRelieFF",
    "FS_Permutation"
  )

colnames(k_time) <- c(
  "PACF@0.01",
  "PACF@0.005",
  "PACF@0.001",
  "PCA",
  "FNN@0",
  "FNN@0.1",
  "FNN@0.01",
  "GridSearch",
  "AIC",
  "BIC",
  "Freq",
  "FS_Permutation",
  "FS_Corr",
  "FS_MI"#,
  #"FS_RRelieFF"
)

perf_ranks <- t(apply(perf, 1, rank))
# distribution of rank per method

bp_dist(perf_ranks)

perf_pd <- perf
perf_pd[] <- lapply(perf_pd,
                  function(x) {
                    percentual_difference(x, perf_pd$`Const-2`)
                  })

bp_dist(log_trans(perf_pd)) +
  labs(y='Perc Diff Error')




#
avg_rank <- colMeans(perf_ranks)
sdev_rank <- apply(perf_ranks, 2, sd)
ord <- order(avg_rank)
avg_rank <- avg_rank[ord]
sdev_rank <- sdev_rank[ord]
avg_rank_plot(avg_rank, sdev_rank, 'Avg Rank')


bp_dist(log(k_time + 1)) +
  labs(y = 'Time execution (in log seconds)')


# distribution of k_hat per method
k_hat[k_hat > 50] <- 50
bp_dist(k_hat) +
  labs(y = 'Embed size distribution')


bp_dist(log_trans(perf_pd[k_hat$FS_Permutation>2,])) +
  labs(y='Perc Diff Error')

bp_dist(perf_ranks[k_hat$FS_Permutation>2,]) +
  labs(y='Perc Diff Error')


rope_<-1
# bayes sign test wrt to baseline: bl2
bayes_s_r <- lapply(perf_pd, function(x) {
  BayesianSignTest(diffVector = x[!is.na(x)], rope = c(-rope_, rope_))
  # BayesianSignedRank(diffVector = x[!is.na(x)],-rope_, rope_)
})

bayes_s_r <- do.call(rbind, bayes_s_r)
bayes_s_r <- as.data.frame(bayes_s_r)
bayes_s_r[] <- lapply(bayes_s_r, as.numeric)

colnames(bayes_s_r) <- c('Const-2 Loses',
                         'Draw',
                         'Const-2 Wins')
bayes_plot(bayes_s_r)




