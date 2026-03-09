import numpy as np
import matplotlib
import matplotlib.cm as cm
import cv2
import math


eps = 0.01

def draw_handpose(canvas, all_hand_peaks, point_size=3):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
            [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]

            # if  (abs(abs(x1)- (1)) > eps and abs(abs(x2)- (1)) > eps and abs(abs(y1)- (1) )> eps and abs( abs(y2)- (1) )> eps ):
            #     pass
            # else:
            #     continue
            x1 = int(round(x1 * W))
            y1 = int(round(y1 * H))
            x2 = int(round(x2 * W))
            y2 = int(round(y2 * H))
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps :
            
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=point_size)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            # if  abs(abs(x)- (1)) > eps and abs(abs(y)- (1) ) > eps :
            #     pass
            # else:
            #     continue
            x = int(round(x * W))
            y = int(round(y * H))
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), point_size, (0, 0, 255), thickness=-1)
                ##cv2.putText(canvas, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    return canvas



def draw_facepose(canvas, all_lmks, point_size=3):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), point_size, (255, 255, 255), thickness=-1)
    return canvas


def draw_bodypose_V2_P1(canvas, candidate, subset, stickwidth=6, point_radius=5):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    # resolution = 512
    # k = float(resolution) / min(H, W)

    # stickwidth = 6

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
            [1, 16], [16, 18], [14, 19], [11, 20]]
                #[3, 17], [6, 18], 


    colors =  [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [85, 85, 255],
            [85, 255, 0], [85, 255, 85], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]]



    for i in range(17 + 2):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index or -3 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    return canvas


def draw_bodypose_V2_P2(canvas, candidate, subset, stickwidth=6, point_radius=5):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    # resolution = 512
    # k = float(resolution) / min(H, W)

    # stickwidth = 6

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
            [1, 16], [16, 18], [14, 19], [11, 20]]
                #[3, 17], [6, 18], 


    colors =  [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [85, 85, 255],
            [85, 255, 0], [85, 255, 85], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]]



    for i in range(18 + 2):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1 or index == -3:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)

            if i < 18:
                color_it = colors[i]
            else:
                color_it = [0,255,0]
            cv2.circle(canvas, (int(x), int(y)), point_radius, colors[i], thickness=-1)
            # cv2.putText(canvas, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

    return canvas


def draw_two_pose(pose_list, canvas, draw_body=True, draw_hand_list=[True, True], draw_face=False, min_idx=1):
    if min_idx == 0:
        body_width = 4
        body_point_radius = 3
        face_point_radius = 2
        hand_point_radius = 2
    elif min_idx == 1:
        body_width = 6
        body_point_radius = 5
        face_point_radius = 3
        hand_point_radius = 3
    elif min_idx == 2:
        body_width = 9
        body_point_radius = 7
        face_point_radius = 4
        hand_point_radius = 4

    for pose, draw_hand in zip(pose_list, draw_hand_list):
        bodies = pose['bodies']
        candidate = bodies['candidate']
        subset = bodies['subset']

        canvas = draw_bodypose_V2_P1(canvas, candidate, subset, stickwidth=body_width, point_radius=body_point_radius)

    canvas = (canvas * 0.6).astype(np.uint8)

    for pose, draw_hand in zip(pose_list, draw_hand_list):
        bodies = pose['bodies']
        candidate = bodies['candidate']
        subset = bodies['subset']

        canvas = draw_bodypose_V2_P2(canvas, candidate, subset, stickwidth=body_width, point_radius=body_point_radius)

    for pose, draw_hand in zip(pose_list, draw_hand_list):
        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']

        if draw_hand:
            canvas = draw_handpose(canvas, hands, point_size=hand_point_radius)
        if draw_face:
            canvas = draw_facepose(canvas, faces, point_size=face_point_radius)

    return canvas


colormap = np.array(cm.get_cmap("inferno").colors)
def draw_lines_depth(img,start,end,depthx,depthy, thickness=6):
    if int(start[0]) ==0 and int(start[1]) ==0:
        return img
    if int(end[0]) ==0 and int(end[1]) ==0:
        return img
    colorx = (colormap[(depthx*255).astype(np.uint8)]*255).astype(np.uint8)
    colory = (colormap[(depthy*255).astype(np.uint8)]*255).astype(np.uint8)
    #print(tuple(colorx.tolist()))
    img = cv2.circle(img, (int(start[0]), int(start[1])), thickness, tuple(colorx.tolist()), thickness=-1)
    
    max_span = np.max(np.abs(start-end))
    span = int(max_span//thickness+1)
    for i in range(span):
        x,y,_ = ((span-i)*start+i*end)//span
        cx = ((span-i)*depthx+i*depthy)/span
        colorcx = (colormap[(cx*255).astype(np.uint8)]*255).astype(np.uint8)
        img = cv2.circle(img, (int(x), int(y)), thickness, tuple(colorcx.tolist()), thickness=-1)
    img = cv2.circle(img, (int(end[0]), int(end[1])), thickness, tuple(colory.tolist()), thickness=-1)
    return img

def bodypose_24to20(dwpose_134, threshold=0.3):
    # dwpose_134 (F, 1, 134, 3)
    # candidate (F, 134, 2)
    # subset (F, 134)

    candidate = dwpose_134[:, 0, :, :2]
    subset = dwpose_134[:, 0, :, 2]

    candidate[:,18] = np.mean(candidate[:, [18, 19]], axis=1)
    candidate[:,19] = np.mean(candidate[:, [21, 22]], axis=1)

    l1 = candidate[:, 18]
    l2 = candidate[:, 19]

    r1 = candidate[:, 21]
    r2 = candidate[:, 22]

    # 判断每个坐标的横纵坐标绝对值是否为 1
    is_inside_l = np.logical_and(np.all(np.abs(l1) < 1, axis=1),
                                np.all(np.abs(l2) < 1, axis=1)).astype(int)
    is_inside_r = np.logical_and(np.all(np.abs(r1) < 1, axis=1),
                                np.all(np.abs(r2) < 1, axis=1)).astype(int)

    left_foottoe_score = np.logical_and(
        subset[:, 18] > threshold,
        subset[:, 19] > threshold).astype(int)


    right_foottoe_score = np.logical_and(
        subset[:, 21] > threshold,
        subset[:, 22] > threshold).astype(int)


    subset[:,18] = 10 * np.logical_and(left_foottoe_score, is_inside_l).astype(int)
    subset[:,19] = 10 * np.logical_and(right_foottoe_score, is_inside_r).astype(int)

    return dwpose_134



def draw_dwpose_2d(dwpose_134_two_persons, width, height, threshold=5.0, min_idx=1, show_toes=True, draw_hand=True, draw_face=False):

    canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    personnumber = dwpose_134_two_persons.shape[0]
    
    draw_hand_list = []
    pose_list = []
    for i in range(personnumber):
        candidate = dwpose_134_two_persons[i:i+1, :, :2] # (1, 134, 2)
        subset = dwpose_134_two_persons[i:i+1, :, 2] # (1, 134)

        # bodypose_24to20(candidate, subset, threshold)

        BODY_NUM = 18
        if show_toes:
            BODY_NUM = 20

        nums, keys, locs = candidate.shape # (1, 134, 2)
        # body = candidate[:,:18].copy()
        # body = body.reshape(nums*18, locs)
        body = candidate[:,:BODY_NUM].copy()
        body = body.reshape(nums*BODY_NUM, locs)
        score = subset[:,:BODY_NUM]

        # if show_toes:
                    
        #     left_foottoe = np.mean(candidate[:, [18, 19]], axis=1)
        #     right_foottoe = np.mean(candidate[:, [21, 22]], axis=1)

        #     l1 = candidate[:, 18]
        #     l2 = candidate[:, 19]

        #     r1 = candidate[:, 21]
        #     r2 = candidate[:, 22]

            
        #     # 判断每个坐标的横纵坐标绝对值是否为 1
        #     is_inside_l = np.logical_and(np.all(np.abs(l1) < 1, axis=1),
        #                                 np.all(np.abs(l2) < 1, axis=1)).astype(int)
        #     is_inside_r = np.logical_and(np.all(np.abs(r1) < 1, axis=1),
        #                                 np.all(np.abs(r2) < 1, axis=1)).astype(int)


        #     left_foottoe_score = np.logical_and(
        #         subset[:, 18] > threshold,
        #         subset[:, 19] > threshold).astype(int)

        #     left_foottoe_score = np.logical_and(left_foottoe_score, is_inside_l).astype(int)
        #     right_foottoe_score = np.logical_and(
        #         subset[:, 21] > threshold,
        #         subset[:, 22] > threshold).astype(int)

        #     right_foottoe_score = np.logical_and(right_foottoe_score, is_inside_r).astype(int)


        for i in range(len(score)):
            for j in range(len(score[i])):

                if j < BODY_NUM:
                    if score[i][j] > threshold:
                        score[i][j] = int((BODY_NUM)*i+j)
                    else:
                        score[i][j] = -1
                # elif j == 18:
                #     if left_foottoe_score[i] > 0: #这里eft_foottoe_score[i]已经是0，1值了:
                #         score[i][j] = int((BODY_NUM)*i+j)
                #     else:
                #         score[i][j] = -1
                # elif j == 19:
                #     if right_foottoe_score[i] > 0:#这里eft_foottoe_score[i]已经是0，1值了:
                #         score[i][j] = int((BODY_NUM)*i+j)
                #     else:
                #         score[i][j] = -1

        un_visible = subset < threshold
        candidate[un_visible] = -1

        foot = candidate[:,18:24]
        faces = candidate[:,24:92]
        hands = candidate[:,92:113]
        hands_2 =  candidate[:,113:]

        hands = np.vstack([hands, hands_2])

        # if show_toes:
        #     body =  np.concatenate((body, left_foottoe, right_foottoe), axis=0)

        #print("body.shape is ", body.shape)
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        draw_hand_list.append(draw_hand)
        pose_list.append(pose)

    canvas = draw_two_pose(pose_list, canvas, draw_body=True, draw_hand_list=draw_hand_list, draw_face=draw_face, min_idx=min_idx)

    return canvas