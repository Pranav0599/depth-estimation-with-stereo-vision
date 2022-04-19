import cv2 
import matplotlib.pyplot as plt
import numpy as np


def detect_features(img1, img2):
    """Method to get the feature matches and keypoints
    """
    sift = cv2.SIFT_create()
    # orb = cv2.ORB_create(nfeatures=10000)

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    # f_matches = matches[:30]
    matched_features_image = cv2.drawMatches(img1,keypoints_1,img2,keypoints_2,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    list_kp1 = [list(keypoints_1[mat.queryIdx].pt) for mat in matches] 
    list_kp2 = [list(keypoints_2[mat.trainIdx].pt) for mat in matches]


    return matched_features_image, list_kp1, list_kp2



def get_T_mats(kp1, kp2):
    """Method to get transformation matricesto find fundamental matrix
    """

    kp1_x = [] ; kp1_y = [] ; kp2_x = [] ; kp2_y = []

    kp1 = np.asarray(kp1)
    kp2 = np.asarray(kp2)
    kp1_x_mean = np.mean(kp1[:,0])    
    kp1_y_mean = np.mean(kp1[:,1])    
    kp2_x_mean = np.mean(kp2[:,0])        
    kp2_y_mean = np.mean(kp2[:,1])
    for i in range(len(kp1)): kp1[i][0] = kp1[i][0] - kp1_x_mean
    for i in range(len(kp1)): kp1[i][1] = kp1[i][1] - kp1_y_mean
    for i in range(len(kp2)): kp2[i][0] = kp2[i][0] - kp2_x_mean
    for i in range(len(kp2)): kp2[i][1] = kp2[i][1] - kp2_y_mean
    kp1_x = np.array(kp1[:,0])
    kp1_y = np.array(kp1[:,1])
    kp2_x = np.array(kp2[:,0])
    kp2_y = np.array(kp2[:,1])
    sum_kp1 = np.sum((kp1)**2, axis = 1)
    sum_kp2 = np.sum((kp2)**2, axis = 1) 
    k_1 = np.sqrt(2.)/np.mean(sum_kp1**(1/2))
    k_2 = np.sqrt(2.)/np.mean(sum_kp2**(1/2))
    s_kp1_1 = np.array([[k_1,0,0],[0,k_1,0],[0,0,1]])
    s_kp1_2 = np.array([[1,0,-kp1_x_mean],[0,1,-kp1_y_mean],[0,0,1]])
    s_kp2_1 = np.array([[k_2,0,0],[0,k_2,0],[0,0,1]])
    s_kp2_2 = np.array([[1,0,-kp2_x_mean],[0,1,-kp2_y_mean],[0,0,1]])
    t_1 = np.dot(s_kp1_1,s_kp1_2)
    t_2 = np.dot(s_kp2_1,s_kp2_2)

    return kp1_x, kp1_y, kp2_x, kp2_y,t_1, t_2, k_1, k_2



 
def get_unnormalized_F(kp_1, kp_2):
    """Method to get the un-normalized Fundamental matrix which conatins noise
    """
    
    kp1_x, kp1_y, kp2_x, kp2_y,t_1, t_2, k_1, k_2 = get_T_mats(kp_1, kp_2)

    x1 = ( (kp1_x).reshape((-1,1)) ) * k_1
    y1 = ( (kp1_y).reshape((-1,1)) ) * k_1
    x2 = ( (kp2_x).reshape((-1,1)) ) * k_2
    y2 = ( (kp2_y).reshape((-1,1)) ) * k_2
    # A matrix
    A = []
    for i in range(x1.shape[0]):
        X1, Y1 = x1[i][0],y1[i][0]
        X2, Y2 = x2[i][0],y2[i][0]
        A.append([X2*X1 , X2*Y1 , X2 , Y2 * X1 , Y2 * Y1 ,  Y2 ,  X1 ,  Y1, 1])
    A = np.array(A)
    U, s, VT = np.linalg.svd(A)
    v = VT.T
    f_val = v[:,-1]
    f_mat = f_val.reshape((3,3))
    Uf, s_, Vf = np.linalg.svd(f_mat)
    #Changing to rank 2
    s_[-1] = 0
    temp_s = np.zeros(shape=(3,3)) 
    temp_s[0][0] = s_[0] 
    temp_s[1][1] = s_[1] 
    temp_s[2][2] = s_[2] 
    f_final = np.dot(Uf , temp_s)
    f_final = np.dot(f_final , Vf)
    unnormalized_F = np.dot(t_2.T , f_final)
    unnormalized_F = np.dot(unnormalized_F , t_1)
    
    unnormalized_F = unnormalized_F/unnormalized_F[-1,-1]
    
    return unnormalized_F



def F_mat_Ransac(img_1_features,img_2_features):
    """Applying RANSAC on the unnormalized F matrix to obtain the best suited fundamental matrix by removing noinse
    """
    N = 2000
    sample = 0
    thresh = 0.05
    inliers_atm = 0
    P = 0.99
    final_F_matrix = []

    while sample < N:
        rand_p1 = [] ; rand_p2 = []
        index = np.random.randint( len(img_1_features) , size = 8)
        for i in index:
            rand_p1.append(img_1_features[i])
            rand_p2.append(img_2_features[i])
        Fundamental = get_unnormalized_F(rand_p1, rand_p2)
        #Hartley's 8 points algorithm
        ones = np.ones((len(img_1_features),1))
        x_1 = np.concatenate((img_1_features,ones),axis=1)
        x_2 = np.concatenate((img_2_features,ones),axis=1)
        line_1 = np.dot(x_1, Fundamental.T)
        line_2 = np.dot(x_2,Fundamental)
        e1 = np.sum(line_2* x_1,axis=1,keepdims=True)**2
        e2 = np.sum(np.hstack((line_1[:, :-1],line_2[:,:-1]))**2,axis=1,keepdims=True)
        error =  e1 / e2 
        inliers = error <= thresh
        inlier_count = np.sum(inliers)
        #Best F
        if inliers_atm <  inlier_count:
            inliers_atm = inlier_count
            good_ones = np.where(inliers == True)
            x_1_pts = np.array(img_1_features)
            x_2_pts = np.array(img_2_features)
            in_points_x1 = x_1_pts[good_ones[0][:]]
            in_points_x2 = x_2_pts[good_ones[0][:]]
            final_F_matrix = Fundamental
        inlier_ratio = inlier_count/len(img_1_features)
        denominator = np.log(1-(inlier_ratio**8))
        numerator = np.log(1-P)
        if denominator == 0: continue
        N =  numerator / denominator
        sample += 1
        
    return final_F_matrix, in_points_x1, in_points_x2



def E_matrix(F_matrix, cam0, cam1):
    """Method to comput the essential matrix
    """
    
    e_mat = np.dot(cam1.T,F_matrix)
    e_mat = np.dot(e_mat,cam0)
    #solving for E using SVD
    Ue, sigma_e, Ve = np.linalg.svd(e_mat)
    sigma_final = np.zeros((3,3))
    for i in range(3):
        sigma_final[i,i] = 1
    sigma_final[-1,-1] = 0
    E_mat = np.dot(Ue,sigma_final)
    E_mat = np.dot(E_mat,Ve)
    
    return E_mat


def get_R_T_mats(E_mat, kp1):
    """Method to get the translation and rotation matrices by decomposing essential matrix
    """

    U, s, Vt = np.linalg.svd(E_mat)
    W = np.array([[0,-1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    C1, C2 = U[:, 2], -U[:, 2]
    R1, R2 = np.dot(U, np.dot(W,Vt)), np.dot(U, np.dot(W.T, Vt))
    camera_poses = [[R1, C1], [R1, C2], [R2, C1], [R2, C2]]
    max_len = 0
    for pose in camera_poses:
        front_points = []        
        for point in kp1:
            # Chirelity check
            x = np.array([point[0], point[1], 1])
            v = x - pose[1]
            condition = np.dot(pose[0][2], v)
            if condition > 0:
                front_points.append(point)    
        if len(front_points) > max_len:
            max_len = len(front_points)
            best_camera_pose =  pose
    
    return best_camera_pose[0], best_camera_pose[1]
