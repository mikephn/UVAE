#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
library(lisi)
name <- args[1]
sourceCsv <- args[2]
targetCsv <- args[3]
perplexity <- strtoi(args[4])
n_types <- length(args) - 4

df <- read.csv(sourceCsv)
col_types_start <- length(colnames(df))-n_types
emb <- df[colnames(df)[1:col_types_start]]
col_types <- colnames(df)[(col_types_start+1):ncol(df)]
lisi_scores <- compute_lisi(emb, df, col_types, perplexity = perplexity)
medians <- apply(lisi_scores, 2, median)

scores <- data.frame(matrix(ncol = (length(col_types) + 1), nrow = 0))
colnames(scores) <- c('name', col_types)
scores[1, 'name'] <- name
for (i in seq_along(col_types)) {
  scores[1, col_types[i]] <- medians[i]
}

if (file.exists(targetCsv)) {
  results <- read.csv(targetCsv)
  results[setdiff(names(scores), names(results))] <- NA
  scores[setdiff(names(results), names(scores))] <- NA
  results <- rbind(results, scores)
} else {
  results <- scores
}
write.csv(results, targetCsv, row.names = FALSE)