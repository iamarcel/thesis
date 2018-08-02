if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dtwclust, rgl, gdata, animation, ggplot2, tikzDevice)

source("gesture-clustering.r")

cluster_gestures()
