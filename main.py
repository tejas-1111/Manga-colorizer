'''
Interactive Manga Colorization.
USAGE: python main.py <input_image_path>

Three windows show up, 
Input window shows the input image and the user scribble are done here
Output window accepts the keyboard inputs and displays the final image
Trackbar window changes the color of the scribble

Select the output window and press '0' for intensity continuous
segmentation and '1' for pattern continuous segmentation
The program shows the boundaries obtained after every 10 iterations
Pressing e here terminates the further executions and end the segmentation procedure
You can also press ctrl-c for ending the segmentation at any moment

After the segments are obtained, select the output window  and press 0 for 
color replacement colorization, 1 for stroke preserving colorization, and 2 for
pattern to shading colorization

-------------------------------
Press 
0 for intensity continuous segmentation
1 for pattern continuous segmentation
'''

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
import cv2
import numpy as np
import sys
from code import Level_set_method

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
current_former_x, current_former_y = -1, -1
ix, iy = -1, -1
img = 0
r = 0
g = 0
b = 0


def nothing(x):
    pass

# mouse callback function


def paint_draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode, r, g, b
    # to see when the user starts scribbling
    if event == cv2.EVENT_LBUTTONDOWN:
        # means that he has started to draw
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(img, (current_former_x, current_former_y),
                         (former_x, former_y), (b, g, r), 5)
                current_former_x = former_x
                current_former_y = former_y
    elif event == cv2.EVENT_LBUTTONUP:
        # we see the last point where the user has stopped the scribbling and take that as the starting point
        drawing = False
        if mode == True:
            cv2.line(img, (current_former_x, current_former_y),
                     (former_x, former_y), (b, g, r), 5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x, former_y


def run():
    global current_former_x, current_former_y, drawing, mode, r, g, b, img

    if len(sys.argv) == 3:
        filename = sys.argv[1]
        config_file = sys.argv[2]
    colimg = cv2.imread(filename)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    output = cv2.imread(filename)
    img = output.copy()
    color = np.zeros((10, 10, 3), np.uint8)
    # create three windows and name them
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)
    # this attaches the mouse callback to paint_draw function so that we know when the user scribbles and where
    # he scribbles
    cv2.setMouseCallback('image', paint_draw)
    # create trackbar in order to select color
    cv2.createTrackbar('R', 'trackbar', 0, 255, nothing)
    cv2.createTrackbar('G', 'trackbar', 0, 255, nothing)
    cv2.createTrackbar('B', 'trackbar', 0, 255, nothing)
    sys.stdin.flush()
    alpha = None
    while(1):
        cv2.imshow('output', output)
        cv2.imshow('image', img)
        cv2.imshow('trackbar', color)
        r = cv2.getTrackbarPos('R', 'trackbar')
        g = cv2.getTrackbarPos('G', 'trackbar')
        b = cv2.getTrackbarPos('B', 'trackbar')
        color[:] = b, g, r
        k = cv2.waitKey(10)
        if k == 27:         # esc to exit
            break

        elif k == ord('0'):
            print("Starting Intensity Continuous Segmentation")
            ls = Level_set_method(image, 1, current_former_y,
                                  current_former_x, config_file)
        elif k == ord('1'):
            print("Starting Pattern Continuous Segmentation")
            ls = Level_set_method(image, 0, current_former_y,
                                  current_former_x, config_file)
        else:
            continue
        ls.obtain_segments()
        colortype = int(input(
            "Press 0 for color replacement,1 for stroke preservation and 2 for color shading : "))
        if colortype == 0:
            print("\nColour Replacement Chosen.")
            result = ls.coloring(colimg, 0, (b, g, r))
        elif colortype == 1:
            print("\nStroke Preservation Chosen.")
            result = ls.coloring(colimg, 1, (b, g, r))
        elif colortype == 2:
            print("\nPattern to Shading Chosen.")
            result = ls.coloring(colimg, 2, (b, g, r))
        else:
            print(
                f"Input {colortype} not valid. It should be one of 0,1,2")
            raise ValueError
        plt.figure()
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    run()
    cv2.destroyAllWindows()
