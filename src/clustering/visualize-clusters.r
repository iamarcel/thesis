if (!require("pacman")) install.packages("pacman")
pacman::p_load(gdata, animation, ggplot2, tikzDevice)

save_histogram <- function(data) {
  tikz('../../img/clustering-results-histogram.pgf', width = 6, height = 3)
  ggplot(data, aes(class)) +
  geom_histogram(stat = "count", bins = 8, fill = "#e6effb", colour = "#1E64C8", size = 1) +
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "#666666"),
        axis.text = element_text(colour = "#666666")) +
  labs(x = "Class", y = "Count")
  dev.off()
}

                                        # User matrix for H36M reference frame
userMatrix <- matrix(c(0.8785309, -0.03752132,  0.4761933,    0, -0.2034607,
                       -0.93134338,  0.3019763,    0, 0.4321747, -0.36218596,
                       -0.8258495, 0, 0.0000000,  0.00000000,  0.0000000,    1), ncol = 4, byrow = TRUE)

                                        # User matrix for NAO reference frame
userMatrix <- matrix(c( 0.6609037,   0.7500464,  0.02490377, 0,
                       -0.1723497,   0.1194006,  0.97775847, 0,
                       0.7304008,  -0.6505048,  0.20818269, 0,
                       0.0000000,   0.0000000,  0.00000000, 1),
                     ncol = 4, byrow = TRUE)

plot_pose <- function(pose) {
                                        # Plots point-based pose (H36M format)

  line_start_indices <- c(1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27)
  line_end_indices <-   c(2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28)
  lines <- gdata::interleave(pose[line_start_indices, ], pose[line_end_indices, ])

  rgl::segments3d(lines,
                  xlab = "x", ylab = "y", zlab = "z",
                  xlim = c(-0.5, 0.5), ylim = c(-0.5, 0.5), zlim = c(-0.5, 0.5),
                  lwd = 8, line_antialias = TRUE)

}

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

save_cluster_centers <- function (center) {
  for (v in 1:length(centers)) {
    center <- get_center(centers, v)
    save_animation(center, v)
  }
}

save_cluster_samples <- function (pc_dtw) {
  cluster_i <- 1
  cluster_poses <- unique(which(pc_dtw@cluster == cluster_i))
  str(cluster_poses)
  cluster_samples <- sample(cluster_poses, 4, replace = FALSE)
  str(cluster_samples)

  for (i in 1:length(cluster_samples)) {
    frames <- poses3d[[cluster_samples[[i]]]]
    save_animation(frames, i)
  }
}
