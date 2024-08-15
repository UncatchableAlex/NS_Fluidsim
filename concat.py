import os
import subprocess
chunk_size = 10
iters = 2620
file_list = f'file_list.txt'
with open(file_list, 'w') as f:
    for start_frame in range(0, iters, chunk_size):
        end_frame = min(start_frame + chunk_size, iters)
        f.write(f"file 'ffmpeg_temp/{start_frame}_{end_frame}.mp4'\n")

# Construct the FFmpeg command
ffmpeg_cmd = [
    'ffmpeg',
    '-f', 'concat',
    '-safe', '0',
    '-i', file_list,
    '-c', 'copy',
    f'wing6.mp4'
]

# Run the FFmpeg command
subprocess.run(ffmpeg_cmd, check=True)
os.remove(file_list)