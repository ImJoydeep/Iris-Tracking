import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
# LEFT EYE​
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
# RIGHT EYE
RIGHT_EYE= [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x , p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            # cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,0,255),1, cv2.LINE_AA)
            # cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,0,255),1, cv2.LINE_AA)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_left, int(l_radius), (2,0,255),2,cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (2,0,255),2,cv2.LINE_AA)
            print(mesh_points)
            
            # for coordinates in floatting points
            # print(results.multi_face_landmarks[0].landmark)

    
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
