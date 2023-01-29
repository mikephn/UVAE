#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

library("cyCombine")
library("magrittr")

uncorrected <- readRDS(args[[1]])
corrected <- readRDS(args[[2]])
uncorrected['batch'] <- corrected['batch']

markers <- colnames(uncorrected)
markers <- markers[!markers %in% c('id', 'sample', 'batch')]

celltype_col <- "som"

# Cluster on corrected data
labels <- corrected[markers] %>%
          cyCombine::create_som(rlen = 10,
                                xdim = 8,
                                ydim = 8,
                                markers = markers)
# Add labels
corrected <- corrected %>%
  dplyr::mutate(som = labels)

# Transfer labels to uncorrected data
uncorrected <- corrected %>%
  dplyr::select(id, all_of(celltype_col)) %>%
  dplyr::left_join(uncorrected, by = "id")

# Evaluation using EMD
emd_val <- uncorrected %>%
      cyCombine::evaluate_emd(corrected,
                              binSize = 0.1,
                              markers = markers,
                              cell_col = celltype_col)

mad_val <- uncorrected %>%
      cyCombine::evaluate_mad(corrected,
                              filter_limit = NULL,
                              markers = markers,
                              cell_col = celltype_col)

results <- data.frame(input=args[[1]], output=args[[2]], emd=emd_val$reduction, mad=mad_val$score)
write.csv(results, args[[3]], row.names=FALSE)

plot_density(uncorrected, corrected, y = 'batch', filename = args[[4]])
