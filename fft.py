#print ("helloWorld")

# main.py
import sys
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

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
        return
    elif mode == "2":
        return
    elif mode == "3":
        return
    elif mode == "4":
        return

def mode1(image_path):
    img = plt.imread(image_path)

    transformed_img = DFT_fast_1d(resizeIMG(img))

    plt.figure("Mode 1")
    plt.subplot(1,2,1), plt.imshow(img)
    plt.subplot(1,2,2), plt.imshow(transformed_img)
    plt.show()
    

def resizeIMG(img):
    height, width = img.shape
    pow2Height = closest_power_of_2(height)
    pow2Width = closest_power_of_2(width)
    newDimension = (pow2Width, pow2Height)
    resizedIMG = cv2.resize(img, newDimension)
    return resizedIMG
    # cv2.imshow("Resized image", resizedIMG)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def closest_power_of_2(num):
    if num == 0:
        return 1
    else:
        return 2**math.ceil(math.log2(num))

def DFT_naive(arr):
    #converting from xn to Xk
    newArr =np.asarray(arr, dtype=complex)
    N = newArr.shape[0]
    output = np.zeros(N, dtype=complex)

    for k in range(N):
        # print("k: " + str(k))
        # sum = 0
        for n in range(N):
            output[k] += newArr[n] * np.exp((-1 * np.pi * 2j * k * n)/N)
        # ouput[k] = sum
    return output

def DFT_inverse(arr, N):
    #converting from Xk to Xn
    #inverse
    newArr =[]
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += (1/N) * arr[n] * np.exp((np.pi * 2j * k * n)/N)
        newArr.append(sum)
    return newArr

def DFT_fast_1d(img):
    newArr =np.asarray(img, dtype=complex)
    N = newArr.shape[0]
    if(N <=16):
        return DFT_naive(newArr)
    else:
        oddNewArr = DFT_fast_1d(newArr[1::2])
        evenNewArr = DFT_fast_1d(newArr[0::2])
        ouput = np.zeros(N, dtype=complex)

        for n in range(N):
            ouput[n] = evenNewArr[n % (N//2)] + [np.exp((-1 * np.pi * 2j * n)/N)] * oddNewArr[n % (N//2)]
            # ouput[n] = evenNewArr[n % (N//2)] + np.exp((-1 * np.pi * 2j * n)/N) * oddNewArr[n % (N//2)] #missing k

        return ouput
    
def DFT_fast_2d(img):
    img = np.asarray(img, dtype=complex)
    height, width = img.shape
    output = np.zeros((height, width), dtype=complex)

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