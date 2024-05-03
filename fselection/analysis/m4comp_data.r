# library(devtools)
# install.packages("https://github.com/carlanetto/M4comp2018/releases/download/0.2.0/M4comp2018_0.2.0.tar.gz",repos=NULL)

library(M4comp2018)

min_len <- 500

is_valid <- sapply(M4, function(x) {
  x$period == 'Daily' && x$n > min_len
})

M4_daily <- M4[is_valid]

url_main <- '/Users/vcerqueira/Dropbox/Research/feature_engineering/fselection/data/m4/'


for (i in 1:length(M4_daily)) {
  x <- M4_daily[[i]]$x
  
  df <- as.data.frame(t(t(x)))
  
  write.csv(df, file = paste0(url_main, 'ds_', i,'.csv'), 
            row.names = F)
}

