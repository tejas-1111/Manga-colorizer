import numpy as np
import cv2

def convolution(self, img1, img2):
        '''
        Performs convoluton on two images
        input: img1, img2 -> 2 np arrays
                img1 is the image
                img2 (k x k) is the mask 
        output: result -> np array (WARN: not uint8, handel appropriately)
        '''
        # Flip the mask horizontally and vertically
        mask = np.flipud(np.fliplr(img2.copy()))
        k = mask.shape[0]

        # Pad the original image with zeros
        padding_len = img2.shape[0]//2
        img = cv2.copyMakeBorder(
            img1,
            padding_len,
            padding_len,
            padding_len,
            padding_len,
            cv2.BORDER_CONSTANT,
            value=0
        )

        # Convolution
        result = np.zeros(img1.shape)
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                result[i, j] = np.sum(img[i:i+k, j:j+k]*mask)
        return result