import numpy as np

def gradient(self, img, mode, outs=False):
        '''
        Returns gradient magnitude array of an image (optional -> magnitude of curvature)
        Can use either of 2 modes
        Sobel (1) -> Uses Sobel kernel
        Scharr (2) -> Uses Scharr kernel
        Inbuilt (3) -> Uses np.gradient

        input: img -> np.array, 
                mode -> 1, 2, 3
                outs -> True or False, if True, also returns magnitude of curvature 
        output: result -> np.array (WARN: not uint8, handle appropriately)
                κ -> magnitude of curvature
        '''
        # Set the gradients depending on the mode
        if mode == 1 or mode == 2:
            if mode == 1:
                Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            elif mode == 2:
                Gx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
                Gy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

            # convolute the image and Gx, image and Gy to get gradients in x and y direction
            gradients_x = self.convolution(img, Gx)
            gradients_y = self.convolution(img, Gy)
        elif mode == 3:
            [gradients_y, gradients_x] = np.gradient(img)
        else:
            print("'mode' parameter can only take values 1,2,3")
            raise ValueError

        # combine gradients_x and gradients_y to obtain the magnitude of the gradient of the image
        result = (gradients_x**2 + gradients_y**2)**0.5
        if not outs:
            return result

        # Calculate the magnitude of curvature
        Normalized_gradients_x = gradients_x/(result+1e-10)
        Normalized_gradients_y = gradients_y/(result+1e-10)
        κ = np.gradient(Normalized_gradients_x)[
            1] + np.gradient(Normalized_gradients_y)[0]
        return result, κ