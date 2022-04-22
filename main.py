import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as skl
from scipy.interpolate import interp1d
from skimage.filters import meijering
#from CubicSpline import my_cubic_interp1d
# from sklearn.metrics import mean_absolute_percentage_error
from CS import CS


def morphology_diff(contrast_green, clahe):

    open1 = cv2.morphologyEx(contrast_green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                             iterations=1)
    close1 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                              iterations=1)

    open2 = cv2.morphologyEx(close1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11)),
                             iterations=1)
    close2 = cv2.morphologyEx(open2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11)),
                              iterations=1)

    open3 = cv2.morphologyEx(close2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (23, 23)),
                             iterations=1)
    close3 = cv2.morphologyEx(open3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25)),
                              iterations=1)
    # shit
    # block_size, C, open_kernel, open_iterations, close_kernel, close_iterations = 67, 8, (35, 35), 6, (15, 15), 4
    # # gray_im = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    # gray_correct = np.array(255 * (contrast_green / 255) ** 1.2, dtype='uint8')
    # gray_equ = cv2.equalizeHist(gray_correct)
    # th = cv2.adaptiveThreshold(gray_equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    # step1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)
    # step2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    # step3 = cv2.morphologyEx(step2, cv2.MORPH_OPEN, open_kernel, iterations=3)
    # step4 = cv2.morphologyEx(step3, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    # make diff between contrast_green & blured vision
    contrast_morph = cv2.subtract(close3, contrast_green)
    #cv2.imshow("fg", clahe.apply(open2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clahe.apply(contrast_morph)


def remove_noise(morph_image):
    ret, thr = cv2.threshold(morph_image, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(morph_image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(thr.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(morph_image, morph_image, mask=mask)
    ret, fin_thr = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    new_img = cv2.erode(fin_thr, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
    #cv2.imshow("fg", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return new_img


def remove_blob(clear_image, org_image):
    fundus_eroded = cv2.bitwise_not(clear_image)
    xmask = np.ones(org_image.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if (shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    #cv2.imshow("fg", blood_vessels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blood_vessels

def detect_vessel(org_image):
    copy_org_image = org_image.copy()
    # make split of red green blue colors
    blue, green, red = cv2.split(org_image)
    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    contrast_green = clahe.apply(green)

    # get image after morph - blured & clahe
    morph_image = morphology_diff(contrast_green, clahe)
    morph_image = cv2.adaptiveThreshold(morph_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

    # remove noise
    clear_image = remove_noise(morph_image)
    # remove blobs
    fin_image = remove_blob(clear_image, org_image)
    skelet = skeleton(fin_image)
    i = 0
    j = 0
    for gr, fin in zip(green, skelet):
        for g, f in zip(gr, fin):
            if (f == 0):
                green[i][j] = 255
            j = j + 1
        j = 0
        i = i + 1
    # return fin_image
    #cv2.imshow("fg", green)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return skelet, fin_image, cv2.merge((blue, green, red))

def skeleton (sk):
    # Step 1: Create an empty skeleton
    size = np.size(sk)
    skel = np.zeros(sk.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    i = 1
    while i <= 10:
        #Step 2: Open the image
        open = cv2.bitwise_not(cv2.morphologyEx(sk, cv2.MORPH_OPEN, element))

        #Step 3: Substract open from the original image
        temp = cv2.subtract(sk, open)

        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(sk, element)
        skel = skl(open, method="lee").astype(np.uint8)
        sk = eroded.copy()
        #cv2.imshow("fg", skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        #if cv2.countNonZero(sk) == 0:
            #break
        i = i + 1

    return skel

if __name__ == "__main__":
    data_catalog = "data"
    raw_catalog = "raw_vessels"
    vessel_catalog = "vessels_images"
    sk_catalog = "skelet_images"
    files_names = [x for x in os.listdir(data_catalog) if os.path.isfile(os.path.join(data_catalog, x))]
    files_names.sort()

    for file_name in files_names:
        out_name = file_name.split('.')[0]
        org_image = cv2.imread(data_catalog + '/' + file_name)
        skele, raw_vessel, vessel_image = detect_vessel(org_image)

        cv2.imwrite(raw_catalog + '/' + out_name + ".JPG", raw_vessel)
        cv2.imwrite(sk_catalog + '/' + out_name + ".JPG", skele)
        cv2.imwrite(vessel_catalog + '/' + out_name + ".JPG", vessel_image)
        skelet = cv2.imread(sk_catalog + '/' + file_name)



        # mouse drawing
        def mouse_drawing(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
                circles.append((x, y))

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", mouse_drawing)

        circles = []

        while True:

            #cv2.cvtColor(skele, org, cv2.COLOR_BGR2GRAY)
            frame = cv2.addWeighted(skelet, 0.7, org_image, 0.3, 0)

            for i in range(len(circles) - 1):
                cv2.circle(frame, circles[i], 1, (0, 255, 0), -1)
                if len(circles) >= 2:
                    cv2.line(frame, circles[i], circles[i + 1], (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord("d"):
                circles = []

        cv2.destroyAllWindows()
        print(circles)
        # with open("C.csv", "a") as file:
        #     print(circles, file=file)
        if circles != []:
            # x = []
            # y = []
            # for i in circles:
            #     x.append(i[0])
            #     y.append(i[1])
            # points = np.array(circles)
            # x = points[:, 0]
            # y = points[:, 1]
            CS(circles)
            #arr = np.arange(np.amin(x), np.amax(x), 0.0001)
            #f = interp1d(x, y, 'cubic')
            # xnew = np.linspace(min(x), max(x), 100000)
            # f = my_cubic_interp1d(xnew, x, y)
            # plt.plot(x, y, 'or', xnew, f)

            #plt.plot(x, y, 'o', xnew, f(xnew), '-r')

            #print("\n3 degree: ", np.polyfit(x, y, 3), "\n4 degree: ", np.polyfit(x, y, 4), "\n5 degree: ", np.polyfit(x, y, 5),)
            plt.show()

        # x1 = np.append(x[-1])
        # f_interp = interp1d(x, y, kind='cubic')
        # print("y = ", f_interp)
        # plt.figure()
        #
        # plt.plot(x, y, 'or', f_interp, "--b")
        #
        #
        # plt.show()







