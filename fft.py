#print ("helloWorld")

# main.py
import sys
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy
import random
import time

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
        mode2(image)
    elif mode == "3":
        mode3(image)
    elif mode == "4":
        mode4(image)

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

def mode2(image_path):
    img = plt.imread(image_path)
    transformed_img = DFT_fast_2d(resizeIMG(img))
    height, width = transformed_img.shape
    freq = 0.10
    height_lower_bound = freq * height
    height_upper_bound = (1-freq) * height
    width_lower_bound = freq * width
    width_upper_bound = (1-freq) * width
    transformed_img[int(height_lower_bound):int(height_upper_bound),:] = 0
    transformed_img[:int(width_lower_bound):int(width_upper_bound)] = 0
    
    denoised_img = DFT_fast_2d_inverse(transformed_img).real

    #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    display, (plt1, plt2) = plt.subplots(1,2)
    display.suptitle("Mode 2 for " + image_path)
    plt1.set_title("Original Image")
    plt1.imshow(img)
    plt2.set_title("Denoised Imaged")
    plt2.imshow(numpy.abs(denoised_img), cmap = 'gray')
    plt.show()

def mode3(image_path):
    img = plt.imread(image_path)
    transformed_img = DFT_fast_2d(resizeIMG(img))
    height, width = transformed_img.shape
    numPixels = height * width
    compressions = [5,25,50,80,95]
    transformed_imgs = []
    for i in compressions:
        compressed_img = compression(transformed_img, i, numPixels)
        inversed_img = DFT_fast_2d_inverse(compressed_img).real
        transformed_imgs.append(inversed_img)
    
    #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    display, plts = plt.subplots(2,3)
    display.suptitle("Mode 2 for " + image_path)
    plts[0, 0].set_title("Original Picture")
    plts[0, 0].imshow(img, cmap = 'gray')
    plts[0, 1].set_title("Compression " + str(compressions[0]))
    plts[0, 1].imshow(transformed_imgs[0], cmap = 'gray')
    plts[0, 2].set_title("Compression " + str(compressions[1]))
    plts[0, 2].imshow(transformed_imgs[1], cmap = 'gray')
    plts[1, 0].set_title("Compression " + str(compressions[2]))
    plts[1, 0].imshow(transformed_imgs[2], cmap = 'gray')
    plts[1, 1].set_title("Compression " + str(compressions[3]))
    plts[1, 1].imshow(transformed_imgs[3], cmap = 'gray')
    plts[1, 2].set_title("Compression " + str(compressions[4]))
    plts[1, 2].imshow(transformed_imgs[4], cmap = 'gray')
    
    plt.show()

def mode4(image_path):
    runs = 10
    tests = []
    # for i in range(5):
    #     tests.append(random.randint(2**(5+i), 2**(6+i)))

    # tests = [numpy.random.random((2 ** 5, 2 ** 5)),
    #          numpy.random.random((2 ** 6, 2 ** 6)),
    #          numpy.random.random((2 ** 7, 2 ** 7)),
    #          numpy.random.random((2 ** 8, 2 ** 8))]

    lower_bound = 2**5
    upper_bound = 2**10

    naive_result = []
    fast_result = []
    # for test in tests:
    while lower_bound <= upper_bound:
        temp_naive = []
        temp_fast = []
        test = numpy.random.rand(lower_bound, lower_bound)
        print(test)
        tests.append(test)
        for i in range(runs):
            start = time.time()
            DFT_naive_2d(test)
            end = time.time()
            temp_naive.append(start-end)

            start = time.time()
            DFT_fast_2d(test)
            end = time.time()
            temp_fast.append(start-end)
        
        print("\nThe naive method's mean: " + str(numpy.mean(temp_naive)))
        print("The naive method's variance: " + str(numpy.var(temp_naive)))
        print("The FFT method's mean: " + str(numpy.mean(temp_fast)))
        print("The FFT method's variance: " + str(numpy.var(temp_fast)))
        naive_result.append(numpy.mean(temp_naive))
        fast_result.append(numpy.mean(temp_fast))
        lower_bound **=2


        
    #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    display, (plt1) = plt.subplots(1,1)
    display.suptitle("Mode 4 for " + image_path)
    plt1.set_title("Mean Graph for naive method VS FFT method")
    xpoints = numpy.array(tests)
    ypoints_naive = numpy.array(naive_result)
    ypoints_fast = numpy.array(fast_result)
    plt1.plot(xpoints, ypoints_naive, xpoints, ypoints_fast)
    plt.show()




def compression(img, compressedNum, numPixels):
    compression_complement = 100 - compressedNum
    lower_bound = numpy.percentile(img, compression_complement//2)
    upper_bound = numpy.percentile(img, 100 - compression_complement//2)
    compressed_img = img * numpy.logical_or(img <= lower_bound, img >= upper_bound)
    print("There are " + str(int(numPixels * (compression_complement / 100))) + " non-zero Fourier coefficients when the compression level is " + str(compressedNum))
    return compressed_img

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

def DFT_naive_1d(array):
    newArray = numpy.asarray(array, dtype=complex)
    N = newArray.shape[0]
    X = numpy.empty((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k, n] = numpy.exp(-2j * numpy.pi * n * k / N)

    output = numpy.dot(X,newArray)

    return output

def DFT_naive_1d_inverse(array):
    newArray = numpy.asarray(array, dtype=complex)
    N = newArray.shape[0]
    x = numpy.empty((N, N), dtype=complex)
    for k in range(N):
        for n in range(N):
            x[k, n] = numpy.exp(2j * numpy.pi * n * k / N)

    output = (1/N) * numpy.dot(x,newArray)

    return output

def DFT_fast_1d(array):
    #https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm
    newArray = numpy.asarray(array, dtype=complex)
    N = newArray.shape[0]

    if(N <= 16):
        return DFT_naive_1d(newArray)
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

def DFT_fast_1d_inverse(array):
    #https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm
    newArray = numpy.asarray(array, dtype=complex)
    N = newArray.shape[0]

    if(N <= 16):
        return DFT_naive_1d_inverse(newArray)
    else:
        evenArray = newArray[0::2]
        oddArray = newArray[1::2]

        evenNewArr = DFT_fast_1d_inverse(evenArray)
        oddNewArr = DFT_fast_1d_inverse(oddArray)
        output = numpy.zeros(N, dtype=complex)

        for n in range(N//2):
            even = evenNewArr[n]
            imaginary_odd = numpy.exp((numpy.pi * 2j * n)/N) * oddNewArr[n]

            output[n] = (1/N) * (even + imaginary_odd)
            output[n + N//2] = (1/N) * (even - imaginary_odd)
            
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

def DFT_naive_2d(img):
    img = numpy.asarray(img, dtype=complex)
    height, width = img.shape
    output = numpy.zeros((height, width), dtype=complex)

    for column in range(width):
        output[:, column] = DFT_naive_1d(img[:,column])

    for row in range(height):
        output[row, :] = DFT_naive_1d(output[row, :])

    return output

def DFT_fast_2d_inverse(img):
    img = numpy.asarray(img, dtype=complex)
    height, width = img.shape
    output = numpy.zeros((height, width), dtype=complex)

    for column in range(width):
        output[:, column] = DFT_fast_1d_inverse(img[:,column])

    for row in range(height):
        output[row, :] = DFT_fast_1d_inverse(output[row, :])

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