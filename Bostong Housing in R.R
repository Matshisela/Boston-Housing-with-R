#Loading the packages
library(mlbench)
library(dplyr)
library(KernelKnn)
library(DAAG)

#Loading the Boston Housing Data using the mlbench package
data("BostonHousing")
bdata <- BostonHousing 

View(bdata)