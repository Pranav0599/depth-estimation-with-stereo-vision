import numpy as np
import cv2
import Caliberation



def get_H_mats(fundamental_mat,key_points1,key_points2):
    """Method to obtaian the H matrices from key features and the fundamental matrix
    """
    
    key_points1 = np.asarray(key_points1)
    key_points2 = np.asarray(key_points2)

    # epipoles of images
    U, dt, VT = np.linalg.svd(fundamental_mat)
    V = VT.T
    s = np.where(dt < 0.00001)
    left_img = V[:,s[0][0]]
    right_img = U[:,s[0][0]]
    left_img = np.reshape(left_img,(left_img.shape[0],1))
    right_img = np.reshape(right_img,(right_img.shape[0],1))
    
    T1 = np.array([[1,0,-(700/2)],[0,1,-(500/2)],[0,0,1]])
    e_final = np.dot(T1,right_img)
    e_final = e_final[:,:]/e_final[-1,:]
    distance = ((e_final[0][0])**(2)+(e_final[1][0])**(2))**(1/2)
    if e_final[0][0] >= 0:
        alpha = 1
    else:
        alpha = -1
    T2 = np.array([[(alpha*e_final[0][0])/distance, (alpha*e_final[1][0])/distance, 0],[-(alpha*e_final[1][0])/distance, (alpha*e_final[0][0])/distance, 0],[0, 0, 1]])
    e_final = np.dot(T2,e_final)
    T3 = np.array([[1, 0, 0],[0, 1, 0],[((-1)/e_final[0][0]), 0, 1]])
    e_final = np.dot(T3,e_final)
    H_2 = np.dot(np.dot(np.linalg.inv(T1),T3),np.dot(T2,T1)) #H2 matrix
    h_ones = np.array([1,1,1])
    h_ones = np.reshape(h_ones,(1,3))
    z = np.array([[0,-left_img[2][0],left_img[1][0]],[left_img[2][0],0,-left_img[0][0]],[-left_img[1][0],left_img[0][0],0]])
    M = np.dot(z,fundamental_mat) + np.dot(left_img,h_ones)
    Homography = np.dot(H_2,M)
    temp = np.ones((key_points1.shape[0],1))
    points_1 = np.concatenate((key_points1,temp), axis = 1)
    points_2 = np.concatenate((key_points2,temp), axis = 1)
    x_1 = np.dot(Homography,points_1.T)
    x_1 = x_1[:,:]/x_1[2,:]
    x_1 = x_1.T
    x_2 = np.dot(H_2,points_2.T)
    x_2 = x_2[:,:]/x_2[2,:]
    x_2 = x_2.T
    x_2_dash = np.reshape(x_2[:,0], (x_2.shape[0],1))
    
    # Least squares method
    lis = list()
    X = x_1
    Y = np.reshape(x_2_dash, (x_2_dash.shape[0], 1))
    #B matrix 
    X_ = np.dot(X.T, X)
    X_total_inv = np.linalg.inv(X_)
    Y_ = np.dot(X.T, Y)
    B_mat = np.dot(X_total_inv, Y_)
    new_y = np.dot(X, B_mat)
    for i in new_y:
        for a in i:
            lis.append(a)
    d_1 = np.array([[B_mat[0][0],B_mat[1][0],B_mat[2][0]],[0,1,0],[0,0,1]])
    H_1 = np.dot(np.dot(d_1,H_2),M) #H1 matrix

    return H_1,H_2



def drawlines(img1src, img2src, lines, pts1, pts2):
    """Function to draw the lines on both the images
    """
    s = img1src.shape
    r = s[0]
    c = s[1]
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        pt1 = [int(pt1[0]),int(pt1[1])]
        pt2 = [int(pt2[0]),int(pt2[1])]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1src, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1src, tuple(pt1), 8, color, -1)
        img2color = cv2.circle(img2src, tuple(pt2), 8, color, -1)
    
    return img1color, img2color
    

