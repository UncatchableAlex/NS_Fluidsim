from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess
import os 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

class NS_Plotter:
    def __init__(self, x_range, particle_positions, V, P, pressure_vmin=-0.6, pressure_vmax=0.5, mag_vmin=0, mag_vmax=12):
        self.pressure_vmin = pressure_vmin
        self.pressure_vmax = pressure_vmax
        self.mag_vmin = mag_vmin
        self.mag_vmax = mag_vmax
        self.x_range = x_range
        self.particle_positions = particle_positions
        self.V = V
        self.P = P
        # Set up the figure and axis
        plt.style.use('dark_background')
        self.fig, (self.mag_ax, self.particle_ax, self.pressure_ax) = plt.subplots(nrows=3, ncols=1, figsize=(4, 14), sharex=True, subplot_kw={'xticks': [], 'yticks': []}, layout='constrained')
        # Define the initial frame
        self.fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=self.mag_vmin, vmax=self.mag_vmax), cmap='viridis'), ax=self.mag_ax, shrink=0.8, orientation='horizontal', pad=0.02)
        self.fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=self.pressure_vmin, vmax=self.pressure_vmax), cmap='viridis'), ax=self.pressure_ax, shrink=0.8, orientation='horizontal', pad=0.02)

    # Function to update the plot
    def update_plot(self, current_frame):
        self.mag_ax.clear()
        self.mag_ax.set_title("Speed (m/s)")
        V_frame = self.V[current_frame]
        V_x, V_y = V_frame[:, :, 0], V_frame[:, :, 1]
        norm = np.sqrt(V_x**2 + V_y**2)  # Calculate the magnitude of the vectors
        # Plot the vector field magnitude
        self.mag_ax.imshow(norm, cmap='viridis', extent=[self.x_range[0], self.x_range[1], self.x_range[0], self.x_range[1]])#, vmin=self.mag_vmin, vmax=self.mag_vmax)
        self.mag_ax.set_xlim(self.x_range[0], self.x_range[1])
        self.mag_ax.set_ylim(self.x_range[0], self.x_range[1])
        self.mag_ax.set_xticks([])
        self.mag_ax.set_yticks([])
        
        self.particle_ax.clear()
        self.particle_ax.set_title("Particle Trace")
        self.particle_ax.set_xlim(self.x_range[0], self.x_range[1])
        self.particle_ax.set_ylim(self.x_range[0], self.x_range[1])
        self.particle_ax.scatter(self.particle_positions[current_frame, :, 1], self.particle_positions[current_frame, :, 0], s=0.002, color='hotpink')
        self.particle_ax.set_xticks([])
        self.particle_ax.set_yticks([])

        self.pressure_ax.clear()
        self.pressure_ax.set_title("Pressure (Pa)")
        self.pressure_ax.imshow(self.P[current_frame], cmap='viridis', extent=[self.x_range[0], self.x_range[1], self.x_range[0], self.x_range[1]])#, vmin=self.pressure_vmin, vmax=self.pressure_vmax)
        self.pressure_ax.set_xlim(self.x_range[0], self.x_range[1])
        self.pressure_ax.set_ylim(self.x_range[0], self.x_range[1])
        self.pressure_ax.set_xticks([])
        self.pressure_ax.set_yticks([])
        self.fig.canvas.draw()


    def render_section(self, start_frame, end_frame):
      #  print(start_frame, end_frame)
        canvas_width, canvas_height = self.fig.canvas.get_width_height()
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-s', '%dx%d' % (canvas_width, canvas_height),
            '-pix_fmt', 'argb',
            '-r', '60',
            '-i', 'pipe:0',
            f'ffmpeg_temp/{start_frame}_{end_frame}.mp4'
        ]
        # Run FFmpeg with input data piped to stdin
        sub_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.update_plot(0)
        for frame in range(start_frame,end_frame): #tqdm(range(10), total=10):
            self.update_plot(frame)
            frame_data = self.fig.canvas.tostring_argb()
            # try to force feed the frame bytes through the stdin. If it doesn't work, print the error
            try:
                sub_process.stdin.write(frame_data)
            except Exception as e:
                print(f"Error writing frame {frame}: {e}")
                stderr_data = sub_process.stderr.read()
                if stderr_data:
                    print(f"FFmpeg stderr: {stderr_data.decode('utf-8')}")
                break

        stdout_data, stderr_data = sub_process.communicate()
        if sub_process.returncode != 0:
            print(f"FFmpeg error: {stderr_data.decode('utf-8')}")
        #else:
          #  print("FFmpeg completed successfully.")


    def render(self, name):
        cores = self.V.shape[0]//5
        exec = ProcessPoolExecutor()
        frames = self.particle_positions.shape[0]
        frames_per_process = (frames // cores) + 1
        processes = []
        try:
            os.mkdir('ffmpeg_temp')
        except:
            pass
        
        for start_frame in range(0, int(cores*frames_per_process), frames_per_process):
            end_frame = min(start_frame + frames_per_process, frames)
            p = exec.submit(self.render_section, start_frame, end_frame)
            processes.append(p)
            print(f'starting {start_frame}')

        for p in processes:
            p.result()
           # print(p.exitcode)


        file_list = f'file_list.txt'
        with open(file_list, 'w') as f:
            for start_frame in range(0, int(cores*frames_per_process), frames_per_process):
                end_frame = min(start_frame + frames_per_process, frames)
                f.write(f"file 'ffmpeg_temp/{start_frame}_{end_frame}.mp4'\n")
        
        # Construct the FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list,
            '-c', 'copy',
            f'{name}.mp4'
        ]

        # Run the FFmpeg command
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(file_list)
        for start_frame in range(0, int(cores*frames_per_process), frames_per_process):
            end_frame = min(start_frame + frames_per_process, frames)
            os.remove(f"ffmpeg_temp/{start_frame}_{end_frame}.mp4")

        os.rmdir('ffmpeg_temp')
