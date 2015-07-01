#!/usr/bin/env Rscript
#
# Converts FlyBase IDs in FlyNet to Entrez IDs

library("org.Dm.eg.db")

netFile <- "/work/jyoung/DataDownload/FunctionalNet/FlyNet.txt"
flyNet <- read.table(netFile)
flybase2entrez <- as.list(org.Dm.egFLYBASE2EG)
flyNet$V1 <- unname(flybase2entrez[as.vector(flyNet$V1)])
flyNet$V2 <- unname(flybase2entrez[as.vector(flyNet$V2)])
flyNet <- as.matrix(flyNet)
writeFile <- "../data/FlyNetEntrez.txt"
write.table(flyNet, file=writeFile, quote=FALSE, sep="\t", 
            row.names=FALSE, col.names=FALSE)

