if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, dtwclust, gdata)

IN_CLIPS_PATH <- "../clips.jsonl"
OUT_CLIPS_PATH <- IN_CLIPS_PATH

cluster_gestures <- function() {
  # Clusters poses and saves the cluster results into the clip file

  data <- jsonlite::stream_in(file(IN_CLIPS_PATH, open = "r"))
  angles <- data[, "angles"]
  angles <- angles[lapply(angles, length) > 0]

  if (file.exists("model.RData")) {
    load(file = "model.RData")
  } else {
    pc_dtw <- tsclust(angles, k = 8L,
                      type = "partitional",
                      distance = "dtw", centroid = "pam",
                      trace = TRUE, seed = 8,
                      norm = "L2", window.size = 40L,
                      args = tsclust_args(cent = list(trace = TRUE)))
    save(pc_dtw, file = "model.RData")
  }

  data["class"] <- pc_dtw@cluster

  con <- file(OUT_CLIPS_PATH, open = "w")
  data <- jsonlite::stream_out(data, con)
  close(con)
}

save_center_angles <- function(centroids) {
  for (i in 1:length(centers)) {
    con <- file(paste("center-", i, ".jsonl", sep = ""), open = "w")
    jsonlite::stream_out(data.frame(centers[[i]]), con)
    close(con)
  }

  list_of_centers <- lapply(pc_dtw@centroids, data.frame)
  jsonlite::write_json(list(clusters=list_of_centers), "../cluster-centers.json")
}

read_center_points <- function() {
  # Reads point-based centroid clusters
  #
  # The angle-based centers exported by the save_center_angles are converted
  # to point-based centers by running
  #
  #   /src/util.py write-poses-from-angle-files clustering/center-points.jsonl clustering/center-*.jsonl
  #
  # Of course, you can plot them with the Python functions, too.

  con <- file("center-points.jsonl", open = "r")
  centers <- jsonlite::stream_in(con)[, "points_3d"]
  close(con)
}
