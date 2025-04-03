from .scene import Scene
import numpy as np
from pathlib import Path


def create_animation(scene,samples_per_pixel,max_depth, fps, start_time, final_time, update_scene, name):

    """
	this function render a list of frames and saves them in ./frames folder. You can make an animation the using ffmpeg running
	from the command prompt:
    """
    #ffmpeg -r 60 -f image2 -s 854x480 -i your_image_%d.png -vcodec libx264 -crf 1 -pix_fmt yuv420p your_video.mp4
              #fps          #resoluion                                      #crf = quality (less is better)


    number_of_frames = int(fps*(final_time - start_time))
    dt = (final_time - start_time)/number_of_frames
    t = start_time

    try:
    	Path("./frames").mkdir()

    except FileExistsError:
        pass


    for i in range(0,number_of_frames):
        print(i)
        update_scene(scene, t)
        img = scene.render(samples_per_pixel, max_depth)
        t += dt
        img.save("frames/" + name + "_" + str(i) + ".png")
