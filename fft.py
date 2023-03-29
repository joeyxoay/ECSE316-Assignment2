#print ("helloWorld")

# main.py
import sys
import math
import cv2


mode = "1"
image = "moonlanding.png"


def parsingArg(argv):
    for i, arg in enumerate(argv):
        #print(f"Argument {i:>6}: {arg}")
        
        if arg.casefold() == "-m":
            mode = argv[i+1]
        elif arg.casefold() == "-i":
            image = argv[i+1]
    
    if mode != "1" and mode != "2" and mode != "3" and mode != "4":
        print("Invalid mode entry")

    # print("mode is " + mode)
    # print("image is " + image)

def modeOption(mode):
    if mode == "1":
        return
    elif mode == "2":
        return
    elif mode == "3":
        return
    elif mode == "4":
        return

def DFFX(arr, N):
    #converting from xn to Xk
    newArr =[]
    for k in range(N-1):
        sum = 0
        for n in range(N-1):
            sum += arr[n] * math.exp((-1 * math.pi * 2j * k * n)/N)
        newArr.append(sum)
    return newArr

def DFFx(arr, N):
    #converting from Xk to Xn
    #inverse
    newArr =[]
    for k in range(N-1):
        sum = 0
        for n in range(N-1):
            sum += (1/N) * arr[n] * math.exp((math.pi * 2j * k * n)/N)
        newArr.append(sum)
    return newArr

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    print(f"Name of the script      : {sys.argv[0]=}")
    print(f"Arguments of the script : {sys.argv[1:]=}")
    parsingArg(sys.argv)

    print("Outside function. mode is " + mode)
    cv2.imread("./moonlanding.png")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")