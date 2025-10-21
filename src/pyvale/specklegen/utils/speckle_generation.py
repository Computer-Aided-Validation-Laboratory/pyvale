import numpy as np
from scipy import ndimage

def pixelsInDisk(cent_x: int, cent_y: int, 
                 screen_size_width: int, screen_size_height: int, 
                 speckle_size: float, foreground_colour: int, image: np.ndarray) -> None:

        box_max_x = int(np.ceil(cent_x + speckle_size / 2))
        box_min_x = int(np.floor(cent_x - speckle_size / 2))
    
        box_max_y = int(np.ceil(cent_y + speckle_size / 2))
        box_min_y = int(np.floor(cent_y - speckle_size / 2))
    
        for yy in range(box_min_y, box_max_y):
            for xx in range(box_min_x, box_max_x):
                if yy < 0 or yy >= screen_size_height or xx < 0 or xx >= screen_size_width:
                    continue
                distance = (xx + 0.5 - cent_x)**2 + (yy + 0.5 - cent_y)**2
                if distance <= (speckle_size / 2)**2:
                    image[yy, xx] = foreground_colour

def generate_speckles(screen_size_width: int, screen_size_height: int, 
                      speckle_size: float, foreground_colour: int,
                      total_speckles: int, reduce_overlap: bool,
                      bit_depth: int, background_colour: int,
                      sigma: float, attempts_tot: int = 100) -> np.ndarray:
     
        image: np.ndarray = np.ones((screen_size_height,screen_size_width),
                                    dtype=np.uint16 if bit_depth == 16 else np.uint8) * background_colour
        
        results = np.zeros([total_speckles, 5])  # speckle number, attempts, with/without/not checked overlap (1/0/2), cent_x, cent_y

        if reduce_overlap:
                for i in range(int(total_speckles)):
                    # reduce overlap by checking if the new speckle overlaps with existing ones
                    over_lap = True
                    attempts = 0
                    # print(f"Placing speckle {i+1}/{total_speckles}")
                    while over_lap and attempts < attempts_tot:
                        # print(f"Attempt {attempts+1} to place speckle {i+1}")
                        over_lap = False
                        cent_x = np.random.uniform(0, screen_size_width)
                        cent_y = np.random.uniform(0, screen_size_height)
            
                        # check colour
                        if image[int(cent_y), int(cent_x)] == foreground_colour:
                            over_lap = True
                            attempts += 1
                            # print(f"Overlap detected at centre ({cent_x}, {cent_y})")
                            continue
            
                        box_max_x = int(np.ceil(cent_x + speckle_size / 2))
                        box_min_x = int(np.floor(cent_x - speckle_size / 2))
            
                        box_max_y = int(np.ceil(cent_y + speckle_size / 2))
                        box_min_y = int(np.floor(cent_y - speckle_size / 2))
            
                        for yy in range(box_min_y, box_max_y):
                            for xx in range(box_min_x, box_max_x):
                                if yy < 0 or yy >= screen_size_height or xx < 0 or xx >= screen_size_width:
                                    continue
                                distance = (xx + 0.5 - cent_x)**2 + (yy + 0.5 - cent_y)**2
                                if distance <= (speckle_size / 2)**2:
                                    if image[yy, xx] == foreground_colour:
                                        over_lap = True
                                        break
                            if over_lap:
                                # print(f"Overlap detected at ({xx}, {yy})")
                                break
                        results[i, 0] = i + 1
                        results[i, 1] = attempts + 1
                        results[i, 2] = 0 if over_lap else 1
                        results[i, 3] = cent_x
                        results[i, 4] = cent_y
                        attempts += 1
                    # if attempts == attempts_tot:
                    #     print("Could not place speckle without overlap, placing anyway.")
                    
                    for yy in range(box_min_y, box_max_y):
                        for xx in range(box_min_x, box_max_x):
                            if yy < 0 or yy >= screen_size_height or xx < 0 or xx >= screen_size_width:
                                continue
                            distance = (xx + 0.5 - cent_x)**2 + (yy + 0.5 - cent_y)**2
                            if distance <= (speckle_size / 2)**2:
                                image[yy, xx] = foreground_colour
                                # print(f"({xx}, {yy}) distance: {distance}")
        else:
            for i in range(int(total_speckles)):
                cent_x = np.random.uniform(0, screen_size_width)
                cent_y = np.random.uniform(0, screen_size_height)
                pixelsInDisk(cent_x, cent_y, 
                        screen_size_width, screen_size_height, 
                        speckle_size, foreground_colour, image)
                results[i, 0] = i + 1
                results[i, 1] = 1
                results[i, 2] = 2
                results[i, 3] = cent_x
                results[i, 4] = cent_y
             
            
        # Apply Gaussian blur
        image_blur = np.copy(image)
        image_blur = ndimage.gaussian_filter(image_blur, sigma=sigma)
        
        return image_blur, results