import numpy as np
import argparse
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from utils.speckle_pattern_diagnostics import speckle_pattern_diagnostics
from utils.speckle_generation import generate_speckles_perlin_noise

def main() -> None:

    parser = argparse.ArgumentParser()

    # Load parameter definitions from the JSON file
    with open('./scripts/configs/config_input_params_spec_perlin.json') as f:
        parameters_input = json.load(f)

    # Merge all dictionaries
    parameters = {**parameters_input}
    
    # Automatically add arguments from the JSON dictionary
    for param, param_args in parameters.items():
        # Convert the "type" and "required" from string to actual Python type
        # Check if "type" and "required" are present in the JSON file
        if "type" in param_args.keys():
            param_args["type"] = eval(param_args["type"])
        if "required" in param_args.keys():
            param_args["required"] = eval(param_args["required"])
        parser.add_argument(param, **param_args)
    
    # Parse arguments
    args = parser.parse_args()
    
    print('Args in simulation:')
    print(args)
    print('')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('')
    
    print('Start')

    # Extract parameteres and revert to default values if not provided by user
    screen_size_width = args.screen_size_width if args.screen_size_width is not None else 500
    screen_size_height = args.screen_size_height if args.screen_size_height is not None else 400
    bit_depth = args.bit_depth if args.bit_depth is not None else 8
    theme = args.theme if args.theme is not None else 'white_on_black'
    octaves = args.octaves if args.octaves is not None else "Not defined"

    type_gen = args.type_gen
    res_width = args.res_width
    res_height = args.res_height
    
    assert bit_depth in [8, 16], "Bit depth should be either 8 or 16."
    assert theme in ['black_on_white', 'white_on_black'], "Theme should be either 'black_on_white' or 'white_on_black'."
    
    if type_gen == "perlin":
        subfolder = f"/{screen_size_width}_{screen_size_height}_{bit_depth}_{theme}_{type_gen}_{res_width}_{res_height}"
    elif type_gen == "fractal":
        subfolder = f"/{screen_size_width}_{screen_size_height}_{bit_depth}_{theme}_{type_gen}_{res_width}_{res_height}_{octaves}"
    print(subfolder)
    save_path = args.output_path + subfolder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dynamic_range: int = 2**bit_depth - 1
    background_colour = 0 if theme == 'white_on_black' else dynamic_range
    foreground_colour = dynamic_range if theme == 'white_on_black' else 0
        
    # Generate speckle pattern
    if type_gen == "perlin":
        image = generate_speckles_perlin_noise(screen_size_width, screen_size_height, 
                                               res_height, res_width,
                                               foreground_colour, bit_depth, background_colour, type_gen)
    elif type_gen == "fractal":
        image = generate_speckles_perlin_noise(screen_size_width, screen_size_height, 
                                               res_height, res_width,
                                               foreground_colour, bit_depth, background_colour, type_gen,
                                               octaves=octaves)
    
    # Diagnostics
    print("")
    print('Starting speckle pattern diagnostics...')
    results = speckle_pattern_diagnostics(image, dynamic_range, save_path)

    # save the diagnostics results
    with open(f"{save_path}/speckle_pattern_diagnostics.json", 'w') as f:
        json.dump(results, f, indent=4)

    ratio = results.get("black_white_ratio", None)
    mean_gradient = results.get("mean_intensity_gradient", None)
    std_dev = results.get("std_dev_irradiance", None)
    avg = results.get("avg_irradiance", None)
    contrast = results.get("contrast", None)
    entropy = results.get("shannon_entropy", None)
    peak_to_mean = results.get("peak_to_mean_ratio", None)
    skew = results.get("skewness", None)
    kurt = results.get("kurtosis", None)
    avg_speckle_size_fwhm = results.get("avg_speckle_size_fwhm", None)
    avg_speckle_size_e2 = results.get("avg_speckle_size_e2", None)
    H_fit_stats = results.get("H_fit_stats", None)
    V_fit_stats = results.get("V_fit_stats", None)

    print("")
    print("Speckle statistics:")

    print(f"Black/White ratio: {np.round(ratio, 3)}")
    print(f"Mean intensity gradient: {np.round(mean_gradient, 3)}")
    print(f"Standard deviation of irradiance values: {np.round(std_dev, 3)}")
    print(f"Average irradiance value: {np.round(avg, 3)}")
    print(f"Contrast (std/mean): {np.round(contrast, 3)}")
    print(f"Skewness: {np.round(skew, 3)}")
    print(f"Kurtosis: {np.round(kurt, 3)}")
    print(f"Shannon entropy: {np.round(entropy, 3)}")
    print(f"Peak to mean ratio: {np.round(peak_to_mean, 3)}")
    print(f"Average speckle size (full width at half maximum): {np.round(avg_speckle_size_fwhm, 3)} pixels")
    print(f"Average speckle size (1/e^2): {np.round(avg_speckle_size_e2, 3)} pixels")
    print(f"R_squared: Horisontal fit: {np.round(H_fit_stats['R_squared'], 3)}, Vertical fit: {np.round(V_fit_stats['R_squared'], 3)}")

    np.save(f"{save_path}/image.npy", image)
    print("")
    print('End :)')
    print("")
    print("")
    print("")


if __name__ == "__main__":
    main()