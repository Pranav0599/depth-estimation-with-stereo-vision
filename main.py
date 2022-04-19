import cv2
import matplotlib.pyplot as plt
import numpy as np
import Caliberation
import Rectification
import Corr_dep


#Main pipeline for all tasks
def pipeline(img1, img2, cam0, cam1, f):

#################################### Caliberation ###################################    
    feature_img, keypoints_1, keypoints_2 = Caliberation.detect_features(img1, img2)
    feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)

    plt.imshow(feature_img)
    plt.show()


    F, pts1, pts2 = Caliberation.F_mat_Ransac(keypoints_1[:50], keypoints_2[:50])
    print("----------------------------------------")

    print("Fundamental matrix: ", F)
    print("----------------------------------------")

    E = Caliberation.E_matrix(F,cam0, cam1)
    print("Essesntial matrix: ", E)
    print("----------------------------------------")


    Rotation,Translation = Caliberation.get_R_T_mats(E, keypoints_1)
    print("Rotation: ", Rotation)
    print("Translation: ", Translation)
    print("----------------------------------------")
######################################################################################

#################################### Rectification ###################################    


    H1, H2 = Rectification.get_H_mats(F, keypoints_1[:50], keypoints_2[:50])
    print("H1 matrix: ", H1)
    print("H2 matrix: ", H2)
    print("----------------------------------------")

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines1.reshape(-1, 3)

    pole_image1,pole_image2  = Rectification.drawlines(img1, img2, lines1, pts1[:50], pts2[:50])
    pole_image1,pole_image2 = Rectification.drawlines(img2, img1, lines2, pts1[:50], pts2[:50])

    rectified_img1 = cv2.warpPerspective(pole_image1, H1, (img1.shape[1],img1.shape[0]))
    rectified_img2 = cv2.warpPerspective(pole_image2, H2, (img1.shape[1],img1.shape[0]))

    rectified_img1 = cv2.cvtColor(rectified_img1, cv2.COLOR_BGR2RGB)
    rectified_img2 = cv2.cvtColor(rectified_img2, cv2.COLOR_BGR2RGB)

        
    fig = plt.figure(figsize=(30, 10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(rectified_img1)
    plt.axis('off')
    plt.title("Rectified Image 1")
    fig.add_subplot(1, 2, 2)
    plt.imshow(rectified_img2)
    plt.axis('off')
    plt.title("Rectified Image 2")
    plt.show()
    print("Please Wait for disparity map....")

######################################################################################

#################################### Correspondance ###################################    


    img_1_copy1 = img1.copy()
    img_2_copy1 = img2.copy()

    rec_nolines_img1 = cv2.warpPerspective(img_1_copy1, H1,  (img1.shape[1],img1.shape[0]))
    rec_nolines_img2 = cv2.warpPerspective(img_2_copy1, H2,  (img1.shape[1],img1.shape[0]))

    rect_img1 = cv2.resize(rec_nolines_img1, (700, 500))
    rect_img2 = cv2.resize(rec_nolines_img2, (700, 500))

    disparity = Corr_dep.correspondence(rect_img1, rect_img2)

    fig2 = plt.figure(figsize=(30, 10))
    fig2.add_subplot(1, 2, 1)
    plt.imshow(disparity, cmap='gray')
    plt.axis('off')
    plt.title('Disparity Map Graysacle')
    fig2.add_subplot(1, 2, 2)
    plt.imshow(disparity, cmap='hot')
    plt.axis('off')
    plt.title('Disparity Map Hot')
    plt.show()
    print("Please Wait for depth map....")

#######################################################################################

######################################## Depth ########################################    


    final_depth = Corr_dep.depth_map(disparity, baseline, f )


    fig2 = plt.figure(figsize=(30, 10))
    fig2.add_subplot(1, 2, 1)
    plt.imshow(final_depth, cmap='gray')
    plt.axis('off')
    plt.title('Depth Map Graysacle')
    fig2.add_subplot(1, 2, 2)
    plt.imshow(final_depth, cmap='hot')
    plt.axis('off')
    plt.title('Depth Map Hot')
    plt.savefig('depth_image.png')
    plt.show()

#######################################################################################


if __name__ == '__main__':

    dataset = int(input("Please enter dataset 1, 2 or 3: "))
    
    img1_1 = cv2.imread('data/curule/im0.png')  
    img2_1 = cv2.imread('data/curule/im1.png') 

    img1_2 = cv2.imread('data/octagon/im0.png')  
    img2_2 = cv2.imread('data/octagon/im1.png') 

    img1_3 = cv2.imread('data/pendulum/im0.png')  
    img2_3 = cv2.imread('data/pendulum/im1.png')
    
    

    

    if dataset == 1:
        cam0 = np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15], [0, 0, 1]])
        cam1 = np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15], [0, 0, 1]])
        f = 1758.23
        doffs=0
        baseline=88.39
        width=1920
        height=1080
        ndisp=220
        vmin=55
        vmax=195

        pipeline(img1_1, img2_1, cam0, cam1, f)

    elif dataset == 2:
        cam0 = np.array([[1742.11, 0, 804.90],[ 0, 1742.11, 541.22], [0, 0, 1]])
        cam1 = np.array([[1742.11, 0, 804.90],[ 0, 1742.11, 541.22], [0, 0, 1]])
        f = 1742.11
        doffs=0
        baseline=221.76
        width=1920
        height=1080
        ndisp=100
        vmin=29
        vmax=61

        pipeline(img1_2, img2_2, cam0, cam1, f)

    elif dataset == 3:
        cam0 = np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22], [0, 0, 1]])
        cam1 = np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22], [0, 0, 1]])
        f = 1729.05
        doffs=0
        baseline=537.75
        width=1920
        height=1080
        ndisp=180
        vmin=25
        vmax=150

        pipeline(img1_3, img2_3, cam0, cam1, f)

    