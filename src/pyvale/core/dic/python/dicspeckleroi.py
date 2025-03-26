"""
================================================================================
pyvale: the python validation engine
license: mit
copyright (c) 2025 the computer aided validation team
================================================================================
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt





class DICRegionOfInterest: 




    def interactive_selector:

        # Load image
        image = cv2.imread("roi.png")  # Replace with actual image path

        if image is None:
            print("Error: Could not load image. Please check the file path.")
            exit()

        # Initialize mask and preview
        mask = reset_mask(image.shape)
        preview_image = image.copy()

        cv2.namedWindow("ROI Selection", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("ROI Selection", 800,800)
        cv2.setMouseCallback("ROI Selection", mouse_callback)

        print("Controls:")
        print("  r - Rectangle mode")
        print("  c - Circle mode")
        print("  p - Polygon mode")
        print("  a - Add mode (green)")
        print("  s - Subtract mode (red)")
        print("  u - Undo last shape")
        print("  x - Reset mask")
        print("  q - Quit and save mask")
        print("Polygon mode:")
        print("  - Click multiple points to draw polygon")
        print("  - Click near the first point to close the polygon")

        update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                shape_mode = DRAW_RECTANGLE
                first_click = False
                polygon_points = []
                update_display()

            elif key == ord('c'):
                shape_mode = DRAW_CIRCLE
                first_click = False
                polygon_points = []
                update_display()

            elif key == ord('p'):
                shape_mode = DRAW_POLYGON
                first_click = False
                polygon_points = []
                update_display()

            elif key == ord('a'):
                operation_mode = ADD_MODE
                update_display()

            elif key == ord('s'):
                operation_mode = SUBTRACT_MODE
                update_display()

            elif key == ord('x'):
                mask = reset_mask(image.shape)
                polygon_points = []
                mask_history.clear()
                update_display()

            elif key == ord('u'):
                # Undo functionality
                if mask_history:
                    mask = mask_history.pop()
                    update_display()

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()



    def plot_mask:



        # Constants
DRAW_RECTANGLE = 0
DRAW_CIRCLE = 1
DRAW_POLYGON = 2
ADD_MODE = 1
SUBTRACT_MODE = 0

# Global variables
first_click = False
start_point = None
current_point = None
polygon_points = []
shape_mode = DRAW_RECTANGLE  # Default shape mode
operation_mode = ADD_MODE    # Default operation mode

# Color constants (BGR format)
COLOR_ADD = (0, 255, 0)      # Green for additions
COLOR_SUBTRACT = (0, 0, 255) # Red for subtractions
COLOR_PREVIEW = (255, 255, 0)  # Yellow for preview

# Create a blank mask
def reset_mask(image_shape):
    return np.zeros(image_shape[:2], dtype=np.uint8)

mask = None
image = None
preview_image = None  # For displaying temporary shapes
mask_history = []  # Store previous mask states for undo

def save_mask_state(current_mask):
    """Save the current mask state to history"""
    # Limit history to last 10 states to prevent excessive memory usage
    mask_history.append(current_mask.copy())
    if len(mask_history) > 10:
        mask_history.pop(0)

def get_operation_color():
    return COLOR_ADD if operation_mode == ADD_MODE else COLOR_SUBTRACT

def is_point_near_start(point, start_point, threshold=20):
    """Check if point is near the start point of the polygon"""
    return np.linalg.norm(np.array(point) - np.array(start_point)) < threshold

def mouse_callback(event, x, y, flags, param):
    global first_click, start_point, current_point, polygon_points, mask, image, preview_image

    # Always update preview image from the original
    if image is not None and mask is not None:
        preview_image = image.copy()

        # Apply existing mask with proper coloring
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 255] = COLOR_ADD
        colored_mask[mask == 0] = (0, 0, 0)  # Transparent
        preview_image = cv2.addWeighted(preview_image, 1.0, colored_mask, 0.4, 0)

    # Polygon drawing mode
    if shape_mode == DRAW_POLYGON:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if this is the first point or if near the start point to close the polygon
            if not polygon_points or (len(polygon_points) > 2 and is_point_near_start((x, y), polygon_points[0])):
                # Close the polygon if near start point
                if len(polygon_points) > 2 and is_point_near_start((x, y), polygon_points[0]):
                    # Finalize polygon
                    temp_mask = mask.copy()
                    poly_points = np.array(polygon_points, np.int32)

                    # Save current state before modification
                    save_mask_state(mask)

                    if operation_mode == ADD_MODE:
                        cv2.fillPoly(temp_mask, [poly_points], 255)
                    else:
                        cv2.fillPoly(temp_mask, [poly_points], 0)

                    mask = temp_mask
                    polygon_points = []
                    update_display()
                else:
                    # Start a new polygon
                    polygon_points = [(x, y)]
            else:
                # Add point to polygon
                polygon_points.append((x, y))

            current_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and polygon_points:
            # Update current point for preview
            current_point = (x, y)

            # Create temporary visualization
            temp_img = preview_image.copy()
            color = get_operation_color()

            # Draw polygon points and lines
            for i in range(len(polygon_points) - 1):
                cv2.line(temp_img, polygon_points[i], polygon_points[i+1], color, 2)

            # Draw line to current point
            if polygon_points:
                cv2.line(temp_img, polygon_points[-1], current_point, color, 2)

            # If more than 2 points, show potential closing line
            if len(polygon_points) > 2:
                cv2.line(temp_img, current_point, polygon_points[0], color, 2)

            cv2.imshow("ROI Selection", temp_img)

    # Rectangle and Circle drawing mode
elif shape_mode in [DRAW_RECTANGLE, DRAW_CIRCLE]:
    if event == cv2.EVENT_LBUTTONDOWN:
        if not first_click:
            first_click = True
                start_point = (x, y)
                current_point = (x, y)
            else:
                # Second click - complete the shape
                first_click = False
                temp_mask = mask.copy()

                # Save current state before modification
                save_mask_state(mask)

                if shape_mode == DRAW_RECTANGLE:
                    if operation_mode == ADD_MODE:
                        cv2.rectangle(temp_mask, start_point, (x, y), 255, -1)
                    else:
                        cv2.rectangle(temp_mask, start_point, (x, y), 0, -1)

                elif shape_mode == DRAW_CIRCLE:
                    diameter_left  = start_point
                    diameter_right = (x, y)
                    radius = int(np.linalg.norm(np.array(diameter_right) - np.array(diameter_left)) / 2)
                    centre = (int(diameter_left[0] + (diameter_right[0] - diameter_left[0])/2), 
                              int(diameter_left[1] + (diameter_right[1] - diameter_left[1])/2))

                    if operation_mode == ADD_MODE:
                        cv2.circle(temp_mask, centre, radius, 255, -1)
                    else:
                        cv2.circle(temp_mask, centre, radius, 0, -1)

                mask = temp_mask
                start_point = None
                current_point = None
                update_display()

        elif event == cv2.EVENT_MOUSEMOVE and first_click:
            current_point = (x, y)
            # Create temporary visualization
            temp_img = preview_image.copy()
            color = get_operation_color()

            if shape_mode == DRAW_RECTANGLE:
                cv2.rectangle(temp_img, start_point, current_point, color, 2)
            elif shape_mode == DRAW_CIRCLE:
                diameter_left  = start_point
                diameter_right = current_point
                radius = int(np.linalg.norm(np.array(diameter_right) - np.array(diameter_left)) / 2)
                centre = (int(diameter_left[0] + (diameter_right[0] - diameter_left[0])/2), 
                          int(diameter_left[1] + (diameter_right[1] - diameter_left[1])/2))
                cv2.circle(temp_img, centre, radius, color, 2)

            cv2.imshow("ROI Selection", temp_img)

def update_display():
    global preview_image

    # Create a color mask for visualization
    colored_mask = np.zeros_like(image)

    # Apply color based on mask values
    colored_mask[mask == 255] = COLOR_ADD  # Green for added regions

    # Combine with original image
    display = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    # Status text
    mode_text = "Rectangle" if shape_mode == DRAW_RECTANGLE else ("Circle" if shape_mode == DRAW_CIRCLE else "Polygon")
    operation_text = "Add" if operation_mode == ADD_MODE else "Subtract"

    # Split text into separate lines
    status_lines = [
            "Modes: ",
            "   a=add, ",
            "   s=subtract, ",
            " ",
            "Shapes: ",
            "   r=rect, ",
            "   c=circle, ",
            "   p=polygon ",
            " ",
            "x=reset, "
            "u=undo, "
            "q=quit",
            f"Mode: {mode_text}",
            f"Operation: {operation_text}",

            ]

    # Start position for the first line
    y_start = 30  # Initial y-coordinate
    line_spacing = 30  # Spacing between lines

    # Draw each line separately
    for i, line in enumerate(status_lines):
        y_position = y_start + i * line_spacing  # Adjust y position for each line
        cv2.putText(display, line, (10, y_position), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Store this as the preview image for future updates
    preview_image = display.copy()

    cv2.imshow("ROI Selection", display)


    # Display final maskp
    roi_mask = mask.astype(bool)
    plt.figure(figsize=(10, 8))
    plt.imshow(roi_mask, cmap='gray')
    plt.title("Final ROI Mask")
    plt.show()

    print("Final mask created")
