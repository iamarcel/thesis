if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dtwclust, rgl, gdata, animation)

data <- jsonlite::stream_in(file("clips.jsonl", open = "r"))

poses3d <- data[201:240, "points_3d"]
poses3d <- poses3d[lapply(poses3d, length) > 0]
test_pose <- poses3d[[1]]

userMatrix <- matrix(c(0.8785309, -0.03752132,  0.4761933,    0, -0.2034607,
                       -0.93134338,  0.3019763,    0, 0.4321747, -0.36218596,
                       -0.8258495, 0, 0.0000000,  0.00000000,  0.0000000,    1), ncol = 4, byrow = TRUE)

pc_dtw <- tsclust(poses3d, k = 8L,
                  distance = "dtw_basic", centroid = "dba",
                  trace = TRUE, seed = 8,
                  norm = "L2", window.size = 20L,
                  args = tsclust_args(cent = list(trace = TRUE)))

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

for (v in 1:length(centers)) {
  center <- get_center(centers, v)
  center_n_time <- dim(center)[[1]]

  saveVideo({
    rgl::open3d(userMatrix = userMatrix, windowRect = c(30, 30, 670, 670))
    rgl::aspect3d(1)
    line_ids <- plot_pose(center[1, , ])

    for (i in 1:center_n_time) {
      rgl.pop(type = "shapes", id = line_ids)
      line_ids <- plot_pose(center[i, , ])
      rgl.snapshot(sprintf(ani.options('img.fmt'), i))
    }
    rgl.close()
  }, img.name = "cluster", use.dev = FALSE,
  ani.opts = "controls,height=640px,autobrowse=FALSE",
  video.name = paste("cluster_", v, ".mp4", sep = ""),
  interval = 1.0 / 25.0)
}
