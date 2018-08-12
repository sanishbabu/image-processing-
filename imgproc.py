from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Imgproc:
    def __init__(self):
        self.original = cv2.imread("F:\image1.jpg")
        self.contrast = cv2.imread("F:\image1.jpg")
        self.shopped = cv2.imread("F:\image1.jpg")
        # convert the images to grayscale
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        self.contrast = cv2.cvtColor(self.contrast, cv2.COLOR_BGR2GRAY)
        self.shopped = cv2.cvtColor(self.shopped, cv2.COLOR_BGR2GRAY)

    def initfig(self):
        # initialize the figure
        fig = plt.figure("Images")
        images = ("Original", self.original), ("Contrast", self.contrast), ("Photoshopped", self.shopped)

        # loop over the images
        for (i, (name, image)) in enumerate(images):
            # show the image
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_title(name)
            plt.imshow(image, cmap=plt.cm.gray)
            plt.axis("off")

        # show the figure
        plt.show()

    def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def compare_images(imageA, imageB, title):
        # compute the mean squared error and structural similarity
        # index for the images
        m = Imgproc.mse(imageA, imageB)
        s = ssim(imageA, imageB)

        # setup the figure
        fig = plt.figure(title)
        plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

        # show first image
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA, cmap=plt.cm.gray)
        plt.axis("off")

        # show the second image
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB, cmap=plt.cm.gray)
        plt.axis("off")

        # show the images
        plt.show()

    def compare(self):
        # compare the images
        Imgproc.compare_images(self.original, self.original, "Original vs. Original")
        Imgproc.compare_images(self.original, self.contrast, "Original vs. Contrast")
        Imgproc.compare_images(self.original, self.shopped, "Original vs. Photoshopped")

obj=Imgproc()
obj.initfig()
obj.compare()