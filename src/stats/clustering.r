if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dtwclust, rgl, gdata, animation, ggplot)

data <- jsonlite::stream_in(file("clips.jsonl", open = "r"))

poses3d <- data[, "points_3d"]
data <- data[lapply(poses3d, length) > 0, ]
poses3d <- poses3d[lapply(poses3d, length) > 0]



                                        # Cluster poses

if (file.exists("model.RData")) {
  load(file = "model.RData")
} else {
  pc_dtw <- tsclust(poses3d, k = 8L,
                    distance = "dtw_basic", centroid = "dba",
                    trace = TRUE, seed = 8,
                    norm = "L2", window.size = 20L,
                    args = tsclust_args(cent = list(trace = TRUE)))
  save(pc_dtw, file = "model.RData")
}

data["cluster"] <- pc_dtw@cluster
ggplot(data, aes(cluster)) + geom_histogram(stat = "count", bins = 8)



                                        # Save data for FastText

fasttext_data = paste(paste("__label__", data$cluster, sep = ""), data$subtitle, sep = " ")
lapply(fasttext_data, write, "fasttext_examples.txt", append = TRUE)



                                        # Visualize clusters

userMatrix <- matrix(c(0.8785309, -0.03752132,  0.4761933,    0, -0.2034607,
                       -0.93134338,  0.3019763,    0, 0.4321747, -0.36218596,
                       -0.8258495, 0, 0.0000000,  0.00000000,  0.0000000,    1), ncol = 4, byrow = TRUE)

plot_pose <- function(pose) {
  line_start_indices <- c(1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27)
  line_end_indices <-   c(2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28)
  lines <- gdata::interleave(pose[line_start_indices, ], pose[line_end_indices, ])

  rgl::segments3d(lines,
                  xlab = "x", ylab = "y", zlab = "z",
                  xlim = c(-0.5, 0.5), ylim = c(-0.5, 0.5), zlim = c(-0.5, 0.5),
                  lwd = 8, line_antialias = TRUE)

}

centers <- pc_dtw@centroids
get_center <- function(centers, i) {
  center <- matrix(centers[[i]], byrow = TRUE)
  center_n_time <- dim(center)[[1]] / 32 / 3
  dim(center) <- c(center_n_time, 32, 3)
  return(center)
}

save_animation <- function(frames, name) {
  frames_n_time <- dim(frames)[[1]]

  saveVideo({
    rgl::open3d(userMatrix = userMatrix, windowRect = c(30, 30, 670, 670))
    rgl::aspect3d(1)
    line_ids <- plot_pose(frames[1, , ])

    for (i in 1:frames_n_time) {
      rgl.pop(type = "shapes", id = line_ids)
      line_ids <- plot_pose(frames[i, , ])
      rgl.snapshot(sprintf(ani.options('img.fmt'), i))
    }
    rgl.close()
  }, img.name = "cluster", use.dev = FALSE,
  ani.opts = "controls,height=640px,autobrowse=FALSE",
  video.name = paste("cluster_", name, ".mp4", sep = ""),
  interval = 1.0 / 25.0)
}

                                        # Plot cluster centers
for (v in 1:length(centers)) {
  center <- get_center(centers, v)
  save_animation(center, v)
}

                                        # Plot cluster samples
cluster_i <- 6
cluster_poses <- match(cluster_i, pc_dtw@cluster)
cluster_samples <- sample(cluster_poses, 4)
str(cluster_samples)

for (i in 1:length(cluster_samples)) {
  frames <- poses3d[[cluster_samples[[i]]]]
  save_animation(frames, i)
}
classifying decoder