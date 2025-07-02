# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pyqtgraph.Qt import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


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
        self.__roi_selected = False

        self.drawing_poly = False
        self.drawing_rect = False
        self.drawing_circle = False
        self.removing_poly = False
        self.removing_rect = False
        self.removing_circle = False

    def interactive_selection(self):
        """
        Interactive GUI to select a region of interest (ROI) in the image using openCV.

        User can select different shapes (rectangle, circle, polygon),
        user has the option to add or subtract areas from mask.
        Allows for undoing and resetting the mask.
        """

        # Set up main window and layout
        self.__roi_selected = True
        app = pg.mkQApp("ROI GUI")
        main_window = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()
        main_window.setLayout(main_layout)
        main_window.resize(1000,1000)

        # Sidebar
        sidebar = QtWidgets.QVBoxLayout()
        btn_add_rect = QtWidgets.QPushButton("Add Rectangle")
        btn_add_circle = QtWidgets.QPushButton("Add Circle")
        btn_add_poly = QtWidgets.QPushButton("Add Polygon")
        btn_sub_rect = QtWidgets.QPushButton("Remove Rectangle")
        btn_sub_circle = QtWidgets.QPushButton("Remove Circle")
        btn_sub_poly = QtWidgets.QPushButton("Remove Polygon")
        btn_undo_prev = QtWidgets.QPushButton("Undo Shape")
        btn_redo_prev = QtWidgets.QPushButton("Redo Shape")

        for btn in [btn_add_rect, btn_add_circle, btn_add_poly, btn_sub_rect, btn_sub_circle, btn_sub_poly, btn_undo_prev, btn_redo_prev]:
            sidebar.addWidget(btn)

        sidebar.addStretch()

        # Graphics view
        graphics_widget = pg.GraphicsLayoutWidget()
        main_view = graphics_widget.addViewBox(lockAspect=True)
        img = pg.ImageItem(self.image)
        main_view.addItem(img)
        main_view.disableAutoRange('xy')
        main_view.autoRange()

        fill_layer = pg.ImageItem()
        fill_layer.setZValue(1)
        main_view.addItem(fill_layer)

        height, width = self.image.shape[:2]
        fill_array = np.zeros((height,width,4), dtype=np.uint8)

        roi_list = []
        add_list = []
        undo_list = []  # Stack of (roi, add_flag) tuples that were undone

        def clear_redo_stack():
            """Clear the redo stack when new shapes are added"""
            nonlocal undo_list
            undo_list = []

        def redraw_fill_layer():
            nonlocal fill_array
            if not roi_list:
                fill_layer.setImage(fill_array)
                return

            mask = np.zeros((height, width), dtype=bool)

            for n, roi in enumerate(roi_list):

                if isinstance(roi, pg.RectROI):
                    pos = roi.pos()
                    size = roi.size()
                    x, y = int(pos[1]), int(pos[0])
                    w, h = int(size[1]), int(size[0])
                    x = max(0, min(x, width))
                    y = max(0, min(y, height))
                    w = max(0, min(w, width - x))
                    h = max(0, min(h, height - y))
                    if w > 0 and h > 0:
                        mask[y:y+h, x:x+w] = add_list[n]

                elif isinstance(roi, pg.CircleROI):
                    pos = roi.pos()
                    size = roi.size()
                    cx, cy = pos[1] + size[1]/2, pos[0] + size[0]/2
                    rx, ry = size[1]/2, size[0]/2
                    y_coords, x_coords = np.ogrid[:height, :width]
                    circle_mask = ((x_coords - cx)/rx)**2 + ((y_coords - cy)/ry)**2 <= 1
                    if add_list[n]:
                        mask |= circle_mask
                    else:
                        mask &= ~circle_mask

                elif isinstance(roi, pg.PolyLineROI):
                    points = roi.getState()['points']
                    pos = roi.pos()
                    if len(points) >= 3:
                        vertices = np.array([(p[1]+pos[1], p[0]+pos[0]) for p in points])
                        path = Path(vertices)
                        x_min, x_max = int(np.floor(vertices[:, 0].min())), int(np.ceil(vertices[:, 0].max()))
                        y_min, y_max = int(np.floor(vertices[:, 1].min())), int(np.ceil(vertices[:, 1].max()))
                        x_min = max(0, x_min)
                        x_max = min(width, x_max)
                        y_min = max(0, y_min)
                        y_max = min(height, y_max)
                        if x_max > x_min and y_max > y_min:
                            xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                            points_grid = np.column_stack((xx.ravel(), yy.ravel()))
                            inside = path.contains_points(points_grid)
                            inside_2d = inside.reshape(y_max - y_min, x_max - x_min)
                            if add_list[n]:
                                mask[y_min:y_max, x_min:x_max] |= inside_2d
                            else:
                                mask[y_min:y_max, x_min:x_max] &= ~inside_2d

            fill_array[:, :, 0] = 0
            fill_array[:, :, 1] = 255
            fill_array[:, :, 2] = 0
            fill_array[:, :, 3] = mask * 80
            fill_layer.setImage(fill_array)

        addpen=pg.mkPen('g', width=4)
        subpen=pg.mkPen('r', width=4)
        handlepen=pg.mkPen('b', width=4)
        hoverpen=pg.mkPen('b', width=4)

        def undo_last():
            if roi_list:
                roi = roi_list.pop()
                add_flag = add_list.pop()
                main_view.removeItem(roi)
                # Store the undone shape in the redo stack
                undo_list.append((roi, add_flag))
                redraw_fill_layer()

        def redo_last():
            if undo_list:
                roi, add_flag = undo_list.pop()
                # Re-add the shape to the main lists and view
                roi_list.append(roi)
                add_list.append(add_flag)
                main_view.addItem(roi)
                # Reconnect the signal handler
                roi.sigRegionChanged.connect(redraw_fill_layer)
                redraw_fill_layer()

        poly_points = []
        add_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('b'))
        sub_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush('r'))
        add_line = pg.PlotDataItem(pen=pg.mkPen('b', width=3))
        sub_line = pg.PlotDataItem(pen=pg.mkPen('r', width=3))
        main_view.addItem(add_scatter)
        main_view.addItem(sub_scatter)
        main_view.addItem(add_line)
        main_view.addItem(sub_line)

        # FIXED: Removed 'self' parameter from nested function definitions
        def start_drawing_rect():
            self.drawing_rect = True
            main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            print("Click to add rect.")

        def finish_drawing_rect():
            if not self.drawing_rect:
                return
            main_view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            clear_redo_stack()  # Clear redo stack when adding new shape
            self.drawing_rect = False

        def start_removing_rect():
            self.removing_rect = True
            main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            print("Click to add rect.")

        def finish_removing_rect():
            if not self.removing_rect:
                return
            main_view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            clear_redo_stack()  # Clear redo stack when adding new shape
            self.removing_rect = False

        def start_drawing_circle():
            self.drawing_circle = True
            main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            print("Click to add circle.")

        def finish_drawing_circle():
            if not self.drawing_circle:
                return
            main_view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            clear_redo_stack()  # Clear redo stack when adding new shape
            self.drawing_circle = False

        def start_removing_circle():
            self.removing_circle = True
            main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            print("Click to add circle.")

        def finish_removing_circle():
            if not self.removing_circle:
                return
            main_view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            clear_redo_stack()  # Clear redo stack when adding new shape
            self.removing_circle = False

        def start_drawing_polygon():
            nonlocal poly_points
            self.drawing_poly = True
            poly_points = []
            add_scatter.setData([], [])
            add_line.setData([], [])
            main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            print("Click to add polygon points. Right-click to finish.")

        def start_removing_poly():
            nonlocal poly_points
            self.removing_poly = True
            poly_points = []
            sub_scatter.setData([], [])
            sub_line.setData([], [])
            main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            print("Click to add polygon points. Right-click to finish.")

        def finish_drawing_poly():
            nonlocal poly_points
            if not self.drawing_poly:
                return
            self.drawing_poly = False

            if len(poly_points) >= 3:
                clear_redo_stack()  # Clear redo stack when adding new shape
                roi = pg.PolyLineROI(poly_points, closed=True, pen=addpen, hoverPen=hoverpen, handlePen=handlepen, handleHoverPen=hoverpen)
                roi_list.append(roi)
                add_list.append(True)
                main_view.addItem(roi)
                for handle in roi.getHandles():
                        handle.radius = 10
                        handle.buildPath()
                        handle.update()
                roi.sigRegionChanged.connect(redraw_fill_layer)
                redraw_fill_layer()
                print("Polygon added.")
            else:
                print("Need at least 3 points.")

            poly_points = []
            add_scatter.setData([], [])
            add_line.setData([], [])
            main_view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        def finish_removing_poly():
            nonlocal poly_points
            if not self.removing_poly:
                return
            self.removing_poly = False

            if len(poly_points) >= 3:
                clear_redo_stack()  # Clear redo stack when adding new shape
                roi = pg.PolyLineROI(poly_points, closed=True, pen=subpen, hoverPen=hoverpen, handlePen=handlepen, handleHoverPen=hoverpen)
                roi_list.append(roi)
                add_list.append(False)
                main_view.addItem(roi)
                for handle in roi.getHandles():
                        handle.radius = 10
                        handle.buildPath()
                        handle.update()
                roi.sigRegionChanged.connect(redraw_fill_layer)
                redraw_fill_layer()
                print("Polygon added.")
            else:
                print("Need at least 3 points.")

            poly_points = []
            sub_scatter.setData([], [])
            sub_line.setData([], [])
            main_view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        def mouse_clicked(event):
            if self.drawing_poly:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.scenePos()
                    if main_view.sceneBoundingRect().contains(pos):
                        mouse_point = main_view.mapSceneToView(pos)
                        poly_points.append([mouse_point.x(), mouse_point.y()])
                        add_scatter.setData([p[0] for p in poly_points], [p[1] for p in poly_points])
                        if len(poly_points) > 1:
                            add_line.setData([p[0] for p in poly_points], [p[1] for p in poly_points])
                elif event.button() == QtCore.Qt.MouseButton.RightButton:
                    finish_drawing_poly()

            if self.removing_poly:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.scenePos()
                    if main_view.sceneBoundingRect().contains(pos):
                        mouse_point = main_view.mapSceneToView(pos)
                        poly_points.append([mouse_point.x(), mouse_point.y()])
                        sub_scatter.setData([p[0] for p in poly_points], [p[1] for p in poly_points])
                        if len(poly_points) > 1:
                            sub_line.setData([p[0] for p in poly_points], [p[1] for p in poly_points])
                elif event.button() == QtCore.Qt.MouseButton.RightButton:
                    finish_removing_poly()

            elif self.drawing_rect:
                print(self.drawing_rect)
                main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.scenePos()
                    start_point = main_view.mapSceneToView(pos)
                    roi = pg.RectROI(start_point, [height/6, width/6], pen=addpen, hoverPen=hoverpen, handlePen=handlepen, handleHoverPen=hoverpen)
                    roi.addScaleHandle([1,0], [0.0,1.0])
                    roi.addScaleHandle([0,1], [1.0,0.0])
                    roi.addScaleHandle([0,0], [1.0,1.0])
                    roi.addTranslateHandle([0.5,0.5])
                    for handle in roi.getHandles():
                        handle.radius = 10
                        handle.buildPath()
                        handle.update()
                    main_view.addItem(roi)
                    roi_list.append(roi)
                    add_list.append(True)
                    roi.sigRegionChanged.connect(redraw_fill_layer)
                    redraw_fill_layer()
                    print("rect added.")
                    finish_drawing_rect()

            elif self.removing_rect:
                main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.scenePos()
                    start_point = main_view.mapSceneToView(pos)
                    roi = pg.RectROI(start_point, [height/6, width/6], pen=subpen, hoverPen=hoverpen, handlePen=handlepen, handleHoverPen=hoverpen)
                    roi.addScaleHandle([1,0], [0.0,1.0])
                    roi.addScaleHandle([0,1], [1.0,0.0])
                    roi.addScaleHandle([0,0], [1.0,1.0])
                    roi.addTranslateHandle([0.5,0.5])
                    for handle in roi.getHandles():
                        handle.radius = 10
                        handle.buildPath()
                        handle.update()
                    main_view.addItem(roi)
                    roi_list.append(roi)
                    add_list.append(False)
                    roi.sigRegionChanged.connect(redraw_fill_layer)
                    redraw_fill_layer()
                    print("rect added.")
                    finish_removing_rect()

            elif self.drawing_circle:
                main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.scenePos()
                    start_point = main_view.mapSceneToView(pos)
                    x = start_point.x()-width/10
                    y = start_point.y()-width/10
                    roi = pg.CircleROI([x,y], radius=width/10, pen=addpen, hoverPen=hoverpen, handlePen=handlepen, handleHoverPen=hoverpen)
                    roi.addTranslateHandle([0.5,0.5])
                    for handle in roi.getHandles():
                        handle.radius = 10
                        handle.buildPath()
                        handle.update()
                    roi_list.append(roi)
                    add_list.append(True)
                    main_view.addItem(roi)
                    roi.sigRegionChanged.connect(redraw_fill_layer)
                    redraw_fill_layer()
                    print("circle added.")
                    finish_drawing_circle()
                        
            elif self.removing_circle:
                main_view.setCursor(QtCore.Qt.CursorShape.CrossCursor)
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    pos = event.scenePos()
                    start_point = main_view.mapSceneToView(pos)
                    x = start_point.x()-width/10
                    y = start_point.y()-width/10
                    roi = pg.CircleROI([x,y], radius=width/10, pen=subpen, hoverPen=hoverpen, handlePen=handlepen, handleHoverPen=hoverpen)
                    roi.addTranslateHandle([0.5,0.5])
                    for handle in roi.getHandles():
                        handle.radius = 10
                        handle.buildPath()
                        handle.update()
                    roi_list.append(roi)
                    add_list.append(False)
                    main_view.addItem(roi)
                    roi.sigRegionChanged.connect(redraw_fill_layer)
                    redraw_fill_layer()
                    print("circle added.")
                    finish_removing_circle()

        main_view.scene().sigMouseClicked.connect(mouse_clicked)

        btn_add_rect.clicked.connect(start_drawing_rect)
        btn_add_circle.clicked.connect(start_drawing_circle)
        btn_add_poly.clicked.connect(start_drawing_polygon)
        btn_sub_rect.clicked.connect(start_removing_rect)
        btn_sub_circle.clicked.connect(start_removing_circle)
        btn_sub_poly.clicked.connect(start_removing_poly)
        btn_undo_prev.clicked.connect(undo_last)
        btn_redo_prev.clicked.connect(redo_last)

        main_layout.addLayout(sidebar)
        main_layout.addWidget(graphics_widget)
        main_window.show()
        pg.exec()


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

    def specific_subset(self, ss_x: int, ss_y: int, ss_size: int ) -> None:

            top    = max(0, ss_y)
            bottom = min(self.image.shape[0],ss_y+ss_size)
            left   = max(0, ss_x)
            right  = min(self.image.shape[1],ss_x+ss_size)

            # Apply the mask in the subset region
            self.mask[top:bottom, left:right] = 255
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


    def roiread(self, filename: str = "./roi.tiff", binary: bool = True) -> None:
        """
        Loads the ROI mask from a binary or text file and stores it in self.mask.

        Args:
            filename (str): The path of the file to load.
            binary (bool): If True, loads from a .npy binary file. If False, loads from a text file.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded data is not a valid mask.
        """
        import os
        import numpy as np

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        if binary:
            self.mask = np.load(filename)
        else:
            self.mask = np.loadtxt(filename, dtype=int, delimiter=' ')

        # Optional: check if the loaded data is a proper binary mask (0s and 1s)
        if not np.isin(self.mask, [0, 1]).all():
            raise ValueError("Loaded ROI mask contains values other than 0 and 1.")

        self.__roi_selected = True


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

