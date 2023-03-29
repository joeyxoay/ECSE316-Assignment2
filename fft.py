#print ("helloWorld")

# main.py
import sys
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy

def parsingArg(argv):
    mode = "1"
    image = "moonlanding.png"
    for i, arg in enumerate(argv):
        if arg.casefold() == "-m":
            mode = argv[i+1]
        elif arg.casefold() == "-i":
            image = argv[i+1]
    
    if mode != "1" and mode != "2" and mode != "3" and mode != "4":
        print("Invalid mode entry")

    return mode, image

def modeOption(mode, image):
    if mode == "1":
        mode1(image)
    elif mode == "2":
        return
    elif mode == "3":
        return
    elif mode == "4":
        return

def mode1(image_path):
    img = plt.imread(image_path)

    transformed_img = DFT_fast_2d(resizeIMG(img))

    #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    display, (plt1, plt2) = plt.subplots(1,2)
    display.suptitle("Mode 1 for " + image_path)
    plt1.set_title("Original Image")
    plt1.imshow(img)
    plt2.set_title("Fourier Transformed Image")
    plt2.imshow(numpy.abs(transformed_img), norm=colors.LogNorm())
    plt.show()

def resizeIMG(img):
    height, width = img.shape
    pow2Height = closest_power_of_2(height)
    pow2Width = closest_power_of_2(width)
    newDimension = (pow2Width, pow2Height)
    resizedIMG = cv2.resize(img, newDimension)
    return resizedIMG


def closest_power_of_2(num):
    if num == 0:
        return 1
    else:
        return 2**math.ceil(math.log2(num))

def DFT_naive(array):
    newArray = numpy.asarray(array, dtype=complex)
    N = newArray.shape[0]
    X = numpy.empty((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k, n] = numpy.exp(-2j * numpy.pi * n * k / N)

    output = numpy.dot(X,newArray)

    return output

def DFT_fast_1d(array):
    #https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm
    newArray = numpy.asarray(array, dtype=complex)
    N = newArray.shape[0]

    if(N <= 16):
        return DFT_naive(newArray)
    else:
        evenArray = newArray[0::2]
        oddArray = newArray[1::2]

        evenNewArr = DFT_fast_1d(evenArray)
        oddNewArr = DFT_fast_1d(oddArray)
        output = numpy.zeros(N, dtype=complex)

        for n in range(N//2):
            even = evenNewArr[n]
            imaginary_odd = numpy.exp((-1 * numpy.pi * 2j * n)/N) * oddNewArr[n]

            output[n] = even + imaginary_odd
            output[n + N//2] = even - imaginary_odd
            
        return output


def DFT_fast_2d(img):
    img = numpy.asarray(img, dtype=complex)
    height, width = img.shape
    output = numpy.zeros((height, width), dtype=complex)

    for column in range(width):
        output[:, column] = DFT_fast_1d(img[:,column])

    for row in range(height):
        output[row, :] = DFT_fast_1d(output[row, :])

    return output
    




if __name__ == "__main__":
    # print(f"Arguments count: {len(sys.argv)}")
    # print(f"Name of the script      : {sys.argv[0]=}")
    # print(f"Arguments of the script : {sys.argv[1:]=}")

    mode, image = parsingArg(sys.argv)
    modeOption(mode, image)
    # arr = [1,2,3,4,5]
    # print(DFFX(arr, 3))


    # print("Outside function. mode is " + mode)