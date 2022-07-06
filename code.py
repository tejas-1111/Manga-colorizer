import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
import json
import scipy.ndimage.filters as filters

# TODO add all required for loops into try/catch blocks


class Level_set_method():
    def __init__(self, img, continuity_type, init_x, init_y, config_file="config.json"):
        self.init_x = init_x
        self.init_y = init_y
        self.continuity_type = continuity_type
        self.img = img

        with open(config_file, "r") as f:
            data = json.load(f)
        self.init_dist = data["init_dist"]
        self.init_value = data["init_value"]
        self.gradient_iterations = data["gradient_iterations"]
        self.integration_iterations = data["integration_iterations"]
        self.window_sz = data["window_sz"]
        self.sigma = data["sigma"]
        self.F_A = data["F_A"]
        self.delta = data["delta"]
        self.dt = data["dt"]
        self.eps = data["eps"]
        self.width = data["width"]
        self.zero_pos = set()

    def convolution(self, img1, img2):
        '''
        Performs convoluton on two images
        input: img1, img2 -> 2 np arrays
        output: result -> np array (WARN: not uint8, handel appropriately)
        '''
        return cv2.filter2D(img1, cv2.CV_8UC1, img2)

    def gradient(self, img, outs=False):
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
        [gradients_y, gradients_x] = np.gradient(img)

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

    # Initialize phi with signed distance from scribble
    def initialize_phi(self):
        w = self.window_sz
        x = self.init_x
        y = self.init_y
        img = self.img

        phi = np.zeros(img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                phi[i, j] = self.init_dist*(abs(x-i) + abs(y-j))
        phi[x-w:x+w, y-w:y+w] = -self.init_value
        return phi

    def visualization(self, phi):
        fig2 = plt.figure(2)
        fig2.clf()
        contours = measure.find_contours(phi, 0)
        ax2 = fig2.add_subplot(111)
        # img_copy = self.img.copy()
        ax2.imshow(self.img, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)

    def generate_gaussian_filter(self, sz, sigma):
        ax = np.linspace(-(sz - 1) / 2., (sz - 1) / 2., sz)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.amax(kernel)
        kernel /= np.sum(kernel)
        return kernel

    # Functions related to pattern continuous segmentation
    def generate_gabor_kernels(self, width_factor=1.5, sizes=[9, 25, 47, 27], orientations=[0, 60, 120, 180, 240, 300]):
        # Increase width_factor to consider narrower region
        # TODO Read width_factor, sizes, orientations from .json
        self.gaborKernels = []

        for k in sizes:
            for theta in orientations:
                kernel = cv2.getGaborKernel(
                    (k, k), 4, theta, 12, 0.5, 0, ktype=cv2.CV_32F)
                self.gaborKernels.append(kernel/width_factor*np.sum(kernel))

    # Calculate H_p
    def calculate_H_p(self):
        print("Calculating Hp")
        W_m_n = []
        window_sz = self.window_sz
        for kernel in self.gaborKernels:
            W_m_n.append(self.convolution(self.img, kernel))
        scribble_W_m_n = [im[self.init_x-window_sz:self.init_x+window_sz,
                             self.init_y-window_sz:self.init_y+window_sz] for im in W_m_n]
        scribble_T = np.vstack([[e.mean(), e.var()] for e in scribble_W_m_n])

        self.H_p = np.zeros(self.img.shape)
        padded_W_m_n = [cv2.copyMakeBorder(
            e, window_sz, window_sz, window_sz, window_sz, cv2.BORDER_CONSTANT, value=255) for e in W_m_n]
        for i in range(window_sz, self.img.shape[0] + window_sz):
            for j in range(window_sz, self.img.shape[1] + window_sz):
                window_W_m_n = [e[i-window_sz:i+window_sz, j -
                                  window_sz:j+window_sz] for e in padded_W_m_n]
                window_T = np.vstack([[e.mean(), e.var()]
                                      for e in window_W_m_n])
                self.H_p[i-window_sz, j-window_sz] = 1 / \
                    (1+np.sum((scribble_T-window_T)**2)**0.5)
        print(np.unique(self.H_p, return_counts=True))
        print("Hp calculated")

    # Calculate H_i
    def calculate_H_i(self):
        print("Calculating Hi")
        sigma = self.sigma
        convoluted_im = filters.gaussian_filter(self.img, sigma)
        grad = self.gradient(convoluted_im)
        self.H_i = 1/(1+grad**2)
        print("Hi calculated")

    # Calculate F_i
    def calculate_F_i(self):
        '''
        delta is between 0 and 1.
        It is scaled from [0,1] to [0,M1-M2].
        '''
        print("Calculating Fi")
        F_A = self.F_A
        delta = self.delta
        sigma = self.sigma
        convoluted_im = filters.gaussian_filter(self.img, sigma)
        grad = self.gradient(convoluted_im)
        m2, m1 = np.amin(grad), np.amax(grad)
        delta *= m1-m2
        R = (grad-m2)/(m1-m2-delta)
        R = 1/(1+np.exp(-R))
        # R *= 1/np.amax(R)
        self.F_i = -F_A * R
        print("Fi calculated")

    def get_band(self, phi):
        zero_pos = np.column_stack(np.where(phi <= 0))
        for r in zero_pos:
            self.zero_pos.add(tuple(r))
        queue = [(x, 0) for x in self.zero_pos]
        v = np.zeros(phi.shape, dtype='bool')
        zero_pos_band = [[], []]
        for x in self.zero_pos:
            zero_pos_band[0].append(x[0])
            zero_pos_band[1].append(x[1])
        zero_pos_band = tuple(zero_pos_band)
        v[zero_pos_band] = True
        while len(queue) > 0:
            rt = queue[0]
            queue = queue[1:]
            d = rt[1]
            if rt[1] >= self.width:
                continue
            x, y = rt[0]
            if x > 1 and not v[x-1, y]:
                v[x-1, y] = True
                queue.append(((x-1, y), d+1))
            if x < phi.shape[0]-1 and not v[x+1, y]:
                v[x+1, y] = True
                queue.append(((x+1, y), d+1))
            if y > 1 and not v[x, y-1]:
                v[x, y-1] = True
                queue.append(((x, y-1), d+1))
            if y < phi.shape[1]-1 and not v[x, y+1]:
                v[x, y+1] = True
                queue.append(((x, y+1), d+1))
        v[zero_pos_band] = False
        return np.where(v == True)

    def update_phi(self, phi):
        new_phi = phi.copy()
        F_A = self.F_A
        eps = self.eps
        dt = self.dt
        band = self.get_band(phi)
        for _ in range(self.integration_iterations):
            grad, curve = self.gradient(new_phi, outs=True)
            if self.continuity_type == 0:     # Pattern continuous
                new_phi[band] -= (self.H_p * (F_A - eps *
                                              curve) * grad * dt)[band]
            elif self.continuity_type == 1:   # Intensity continuous
                new_phi[band] -= (self.H_i * (F_A - eps *
                                              curve + self.F_i) * grad * dt)[band]
        return new_phi

    def obtain_segments(self):
        phi = self.initialize_phi()
        if self.continuity_type == 0:     # Pattern continuous
            self.generate_gabor_kernels()
            self.calculate_H_p()
        elif self.continuity_type == 1:   # Intensity continuous
            self.calculate_H_i()
            self.calculate_F_i()
        for i in range(self.gradient_iterations):
            phi = self.update_phi(phi)
            boundary = self.visualization(phi)
            k = cv2.waitKey(0)
            if k == ord('e'):
                break
            plt.pause(1)
            print(f"Gradient Iteration {i+1}")
        self.phi = phi
        # Visualization and return boundary here

    def color_replacement(self, img, color):
        # img here is colored
        result = img.copy()
        result[self.phi < 0] = color
        return result

    def stroke_preserving(self, img, color):
        intensity = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensity3d = np.dstack((intensity, intensity, intensity))
        colored_img = np.zeros(img.shape, dtype=img.dtype)
        colored_img[self.phi < 0] = color
        YUV_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(YUV_img)
        self.calculate_H_i()
        Y_new = self.convolution(Y, (1-self.H_i)**2)
        # Y_new = (Y * (1-self.H_i)**2).astype(Y.dtype)
        colored_YUV = cv2.merge((Y_new, U, V))
        colored_img = cv2.cvtColor(colored_YUV, cv2.COLOR_YUV2BGR)
        colored_img = colored_img * np.round(intensity3d/10).astype(img.dtype)
        result = np.zeros(img.shape, dtype=np.uint8)
        result[self.phi > 0] = colored_img[self.phi > 0] + img[self.phi > 0]
        result[self.phi < 0] = colored_img[self.phi < 0]
        return result

    def pattern_to_shading(self, img, color):
        colored_img = img.copy()
        colored_img[self.phi < 0] = color
        YUV_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(YUV_img)
        kernel = np.ones((self.window_sz, self.window_sz))
        s = self.convolution(Y, kernel)
        s = s/np.amax(s)
        Y_new = (s*Y).astype(Y.dtype)
        print(Y.shape, Y_new.shape)
        print(np.unique(Y_new-Y, return_counts=True))
        result_YUV = cv2.merge((Y_new, U, V))
        result = cv2.cvtColor(result_YUV, cv2.COLOR_YUV2BGR)
        return result

    def coloring(self, img, option, color):
        if option == 0:
            return self.color_replacement(img, color)
        elif option == 1:
            return self.stroke_preserving(img, color)
        elif option == 2:
            return self.pattern_to_shading(img, color)
        else:
            print("'option' parameter can only take values 0,1,2")
            raise ValueError
