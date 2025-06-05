# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


import cv2
import numpy as np
import matplotlib.pyplot as plt





class DICRegionOfInterest:
    """
    A class for interactively selecting and manipulating ROI of an image before passing to the DIC engine. 

    Users can:.
    - Interactively select rectangular, circular, or polygonal regions on the image.
    - Add or subtract selected regions from the mask.
    - Undo and reset the mask changes.
    - Display/save the image with ROI overlayed
    - Programatically select a rectangular ROI for consistency.
    
    Public attributes:
        image (np.ndarray): The image on which regions of interest are selected.
        mask (np.ndarray): A binary mask representing the selected regions of interest.
    """

    def __init__(self, image):
        """
        Initializes the DICRegionOfInterest class with an image.
        
        Args:
            image (str or np.ndarray): Can be a path to an image file or an image array.
            
        Raises:
            ValueError: If the image cannot be loaded or is invalid.
        """

        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image.copy()

        if self.image is None:
            raise ValueError("Invalid image input")

        self.mask = np.zeros(self.image.shape[:2], dtype=bool)

        # Private vars. 
        self.__roi_selected = False
        self.__DRAW_RECTANGLE = 0
        self.__DRAW_CIRCLE = 1
        self.__DRAW_POLYGON = 2
        self.__ADD_MODE = 1
        self.__SUBTRACT_MODE = 0
        self.__polygon_points = []
        self.__first_click = False
        self.__start_point = None
        self.__current_point = None
        self.__shape_mode = self.__DRAW_RECTANGLE # default mode
        self.__operation_mode = self.__ADD_MODE # default mode
        self.__COLOR_ADD = (0, 255, 0) # green
        self.__COLOR_SUBTRACT = (0, 0, 255) # red
        self.__mask_history = []
        self.__preview_image = None

    def interactive_selection(self):
        """
        Interactive GUI to select a region of interest (ROI) in the image using openCV.
        
        User can select different shapes (rectangle, circle, polygon),
        user has the option to add or subtract areas from mask.
        Allows for undoing and resetting the mask.
        """

        cv2.namedWindow("ROI Selection", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("ROI Selection", 800,800)
        cv2.setMouseCallback("ROI Selection", self.__mouse_callback)
        self.__update_display()
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.__shape_mode = self.__DRAW_RECTANGLE
                self.__first_click = False
                self.__polygon_points = []
                self.__update_display()
                
            elif key == ord('c'):
                self.__shape_mode = self.__DRAW_CIRCLE
                self.__first_click = False
                self.__polygon_points = []
                self.__update_display()

            elif key == ord('p'):
                self.__shape_mode = self.__DRAW_POLYGON
                self.__first_click = False
                self.__polygon_points = []
                self.__update_display()

            elif key == ord('a'):
                self.__operation_mode = self.__ADD_MODE
                self.__update_display()

            elif key == ord('s'):
                self.__operation_mode = self.__SUBTRACT_MODE
                self.__update_display()

            elif key == ord('x'):
                self.mask = self.reset_mask(self.image.shape)
                self.__polygon_points = []
                self.__mask_history.clear()
                self.__update_display()

            elif key == ord('u'):
                # Undo functionality
                if self.__mask_history:
                    self.mask = self.__mask_history.pop()
                    self.__update_display()

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        self.__roi_selected = True

    def reset_mask(self, image_shape: np.ndarray) -> np.ndarray:
        return np.zeros(image_shape[:2], dtype=bool)

    def rect_boundary(self, left: int, right: int, top: int, bottom: int) -> None:
        """
        Defines a rectangular region of interest (ROI) by setting a rectangular mask.
        
        Args:
            left (int): Left coordinate of the rectangle.
            right (int): Right coordinate of the rectangle.
            top (int): Top coordinate of the rectangle.
            bottom (int): Bottom coordinate of the rectangle.
        """
        self.mask = self.reset_mask(self.image.shape)
        self.mask[bottom:(self.image.shape[0]-top), left:(self.image.shape[1])-right] = 255
        self.__roi_selected = True

    def imsave(self, filename: str="./roi.tiff") -> None:
        """
        Saves the image with the mask overlayed.
        
        Args:
            filename (str): The path where the result image will be saved.
        
        Raises:
            ValueError: If no ROI is selected.
        """
        if not self.__roi_selected:
            raise ValueError("No ROI selected with \'interactive_selection\' or \'rect_boundary\' ")
        overlay = self.image.copy()
        overlay[self.mask] = (0, 255, 0)
        result = cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(filename, result)

    def roisave(self, filename: str="./roi.tiff", binary: bool=True) -> None:
        """
        Saves the roi as a binary mask or text file.
        
        Args:
            filename (str): The path where the result image will be saved.
            binary (bool): If True, saves as a binary mask. If False, saves as a text file.
        Raises:
            ValueError: If no ROI is selected.
        """
        if not self.__roi_selected:
            raise ValueError("No ROI selected with \'interactive_selection\' or \'rect_boundary\' ")
        
        if binary:
            np.save(filename, self.mask)
        else:
            np.savetxt(filename, self.mask, fmt='%d', delimiter=' ')




    def roiread(self, filename: str="./roi.tiff") -> None:
        """
        Saves the image with the mask overlayed.
        
        Args:
            filename (str): The path where the result image will be saved.
        
        Raises:
            ValueError: If no ROI is selected.
        """
        if not self.__roi_selected:
            raise ValueError("No ROI selected with \'interactive_selection\' or \'rect_boundary\' ")
        overlay = self.image.copy()
        overlay[self.mask] = (0, 255, 0)
        result = cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(filename, result)

    def imshow(self) -> None:
        """
        Displays the current mask in grayscale.
        
        Raises:
            ValueError: If no ROI is selected.
        """
        if not self.__roi_selected:
            raise ValueError("No ROI selected with \'interactive_selection\' or \'rect_boundary\' ")
        plt.imshow((self.mask.astype(np.uint8)) * 255, cmap='gray')
        plt.title("ROI Mask")
        plt.show()



    def __is_point_near_start(self, point: int, start_point: int, threshold: int=20) -> bool:
        """
        Checks if a given point is near the starting point of a polygon.
            
        Args:
            point (tuple): The point to check.
            start_point (tuple): The starting point of the polygon.
            threshold (int, optional): The distance threshold to consider a point near the start. Defaults to 20.

        Returns:
            bool: True if the point is within the threshold distance from the start point, otherwise False.
        """
        return np.linalg.norm(np.array(point) - np.array(start_point)) < threshold



    def __get_operation_color(self) -> None:
        """
        Returns the color to be used for the current operation (green for add, red for subtract).
        
        Returns:
            tuple: A color represented as (B, G, R) for the current operation.
        """
        return self.__COLOR_ADD if self.__operation_mode == self.__ADD_MODE else self.__COLOR_SUBTRACT


    def __save_mask_state(self, current_mask: np.ndarray) -> None:
        """
        Saves the current mask state to the history for potential undo.
        
        Args:
            current_mask (np.ndarray): The current mask to save.
        """
        self.__mask_history.append(current_mask.copy())
        if len(self.__mask_history) > 10:
            self.__mask_history.pop(0)



    def __update_display(self) -> None:
        """
        Updates the display window with the current image, mask, and operation details.

        Displays the current shape mode, operation mode, and allows visualization of the selected ROI.
        """

        # create colour mask for after defining shape
        colored_mask = np.zeros_like(self.image)

        # Apply color based on mask values
        colored_mask[self.mask] = self.__COLOR_ADD  # Green for added regions

        # Combine with original image
        display = cv2.addWeighted(self.image, 0.7, colored_mask, 0.3, 0)

        # Status text
        mode_text = "Rectangle" if self.__shape_mode == self.__DRAW_RECTANGLE else ("Circle" if self.__shape_mode == self.__DRAW_CIRCLE else "Polygon")
        operation_text = "Add" if self.__operation_mode == self.__ADD_MODE else "Subtract"

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
        self.__preview_image = display.copy()

        cv2.imshow("ROI Selection", display)



    def __mouse_callback(self, event, x, y, flags, params):
        """
        Handles mouse events for drawing the ROI (rectangle, circle, polygon) and updating roi mask.
        
        Args:
            event (int): mouse event type.
            x (int): x-coordinate of mouse event.
            y (int): y-coordinate of mouse event.
            flags (int): Additional flags (Not required).
            params (any): Additional parameters (Not requried).
        """
        
       # Always update preview image from the original
        if self.image is not None and self.mask is not None:
            self.__preview_image = self.image.copy()
            
            # Apply existing mask with proper coloring
            colored_mask = np.zeros_like(self.image)
            colored_mask[self.mask] = self.__COLOR_ADD
            self.__preview_image = cv2.addWeighted(self.__preview_image, 0.7, colored_mask, 0.4, 0)

        # Polygon drawing mode
        if self.__shape_mode == self.__DRAW_POLYGON:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if click is near shape start
                if not self.__polygon_points or (len(self.__polygon_points) > 2 and self.__is_point_near_start((x, y), self.__polygon_points[0])):
                    # finish polygon if near start
                    if len(self.__polygon_points) > 2 and self.__is_point_near_start((x, y), self.__polygon_points[0]):

                        # save mask state
                        temp_mask = (self.mask.astype(np.uint8)).copy() * 255
                        poly_points = np.array(self.__polygon_points, np.int32)                        
                        self.__save_mask_state(self.mask)

                        # Add or subtract mask based on mode
                        if self.__operation_mode == self.__ADD_MODE:
                            cv2.fillPoly(temp_mask, [poly_points], 255)
                        else:
                            cv2.fillPoly(temp_mask, [poly_points], 0)

                        self.mask = temp_mask == 255
                        self.__polygon_points = []
                        self.__update_display()
                    else:
                        # Start a new polygon
                        self.__polygon_points = [(x, y)]
                else:
                    # Add point to polygon
                    self.__polygon_points.append((x, y))

                self.__current_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE and self.__polygon_points:
                # Update current point for preview
                self.__current_point = (x, y)

                # Create temporary visualization
                temp_img = self.__preview_image.copy()
                color = self.__get_operation_color()

                # Draw polygon points and lines
                for i in range(len(self.__polygon_points) - 1):
                    cv2.line(temp_img, self.__polygon_points[i], self.__polygon_points[i+1], color, 2)

                # Draw line to current point
                if self.__polygon_points:
                    cv2.line(temp_img, self.__polygon_points[-1], self.__current_point, color, 2)

                # If more than 2 points, show potential closing line
                if len(self.__polygon_points) > 2:
                    cv2.line(temp_img, self.__current_point, self.__polygon_points[0], color, 2)

                cv2.imshow("ROI Selection", temp_img)

        # Rectangle and Circle drawing mode
        elif self.__shape_mode in [self.__DRAW_RECTANGLE, self.__DRAW_CIRCLE]:
            if event == cv2.EVENT_LBUTTONDOWN:
                if not self.__first_click:
                    self.__first_click = True
                    self.__start_point = (x, y)
                    self.__current_point = (x, y)
                else:
                    # complete shape with second click
                    self.__first_click = False
                    temp_mask = (self.mask.astype(np.uint8)).copy() * 255

                    # Save current state before modification (incase user wants to undo)
                    self.__save_mask_state(self.mask)

                    if self.__shape_mode == self.__DRAW_RECTANGLE:
                        if self.__operation_mode == self.__ADD_MODE:
                            cv2.rectangle(temp_mask, self.__start_point, (x, y), 255, -1)
                        else:
                            cv2.rectangle(temp_mask, self.__start_point, (x, y), 0, -1)
  
                    elif  self.__shape_mode == self.__DRAW_CIRCLE:
                        
                        #draw circle based on diameter values.
                        diameter_left  = self.__start_point
                        diameter_right = (x, y)
                        radius = int(np.linalg.norm(np.array(diameter_right) - np.array(diameter_left)) / 2)
                        centre = (int(diameter_left[0] + (diameter_right[0] - diameter_left[0])/2), 
                                int(diameter_left[1] + (diameter_right[1] - diameter_left[1])/2))

                        if  self.__operation_mode == self.__ADD_MODE:
                            cv2.circle(temp_mask, centre, radius, 255, -1)
                        else:
                            cv2.circle(temp_mask, centre, radius, 0, -1)

                    self.mask = temp_mask == 255
                    self.__start_point = None
                    self.__current_point = None
                    self.__update_display()

            elif event == cv2.EVENT_MOUSEMOVE and self.__first_click:
                self.__current_point = (x, y)
                # Create temporary visualization
                temp_img = self.__preview_image.copy()
                color = self.__get_operation_color()

                if  self.__shape_mode == self.__DRAW_RECTANGLE:
                    cv2.rectangle(temp_img, self.__start_point, self.__current_point, color, 2)

                elif self.__shape_mode == self.__DRAW_CIRCLE:
                    diameter_left  = self.__start_point
                    diameter_right = self.__current_point
                    radius = int(np.linalg.norm(np.array(diameter_right) - np.array(diameter_left)) / 2)
                    centre = (int(diameter_left[0] + (diameter_right[0] - diameter_left[0])/2), 
                            int(diameter_left[1] + (diameter_right[1] - diameter_left[1])/2))
                    cv2.circle(temp_img, centre, radius, color, 2)

                cv2.imshow("ROI Selection", temp_img)
