ffmpeg -i cluster_1.mp4 -i cluster_2.mp4 -i cluster_3.mp4 -i cluster_4.mp4 -filter_complex "[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]" -map "[v]" output.mp4
