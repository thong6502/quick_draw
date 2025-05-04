import cv2
import numpy as np
import mediapipe as mp
import os
import math
import torch
from src.config import CLASSES
from src.model import QuickDrawModel
from src.utils import get_overlay, get_images
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, default="checkpoint/best_model.pth", help="model path")
    parser.add_argument("--header-path", "-hp", type=str, default="header", help="path to header images")
    parser.add_argument("--image-path", "-ip", type=str, default="images", help="path to image")
    parser.add_argument("--brush-thickness", "-b", type=int, default=15, help="Brush thickness")
    parser.add_argument("--eraser-thickness", "-e", type=int, default=30, help="Eraser thickness")
    return parser.parse_args()

def check_finger_up(hand_landmarks) -> list[int, int, int, int, int]:
    if not hand_landmarks:
        return [0, 0, 0, 0, 0]
    
    finger_up = []
    # Kiểm tra ngón cái theo trục x
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
        finger_up.append(1)
    else:
        finger_up.append(0)

    tips = [8, 12, 16, 20]
    dips = [7, 11, 15, 19]

    # Kiểm tra 4 ngón còn lại theo trục y
    for tip, dip in zip(tips, dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            finger_up.append(1)
        else:
            finger_up.append(0)
    
    return finger_up


def get_pos(pos_header_item) -> list[int, int, int, int]:
    xmin = pos_header_item[0] - pos_header_item[2]
    ymin = pos_header_item[1] - pos_header_item[3]
    xmax = pos_header_item[0]
    ymax = pos_header_item[1]
    return xmin, ymin, xmax, ymax

def main(args):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ########################################
    # Constants for drawing tools
    BRUSH_THICKNESS = args.brush_thickness
    ERASER_THICKNESS = args.eraser_thickness
    WIDTH = 1280
    HEIGHT = 720
    ########################################

    previous_frame_time = 0
    x1, y1 = 0, 0
    is_predict = False

    detail_header_items = {
        # pos: [right_x, bottom_y, width, height] 
        "paint_black": {"pos": [177, 126, 108, 127], "color": (33, 3, 20), "file": "1.png"},
        "paint_yellow": {"pos": [395, 126, 108, 127], "color": (61, 183, 244), "file": "2.png"}, 
        "paint_red": {"pos": [614, 126, 108, 127], "color": (49, 49, 255), "file": "3.png"},   
        "paint_blue": {"pos": [832, 126, 108, 127], "color": (255, 182, 56), "file": "4.png"},  
        "paint_green": {"pos": [1050, 126, 108, 127], "color": (99, 191, 0), "file": "5.png"},
        "eraser": {"pos": [1223, 126, 108, 127], "color": (0, 0, 0), "file": "6.png"}
    }

    # Load model
    model = QuickDrawModel(len(CLASSES))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # Load header images
    folderPath = args.header_path
    overlaylist = []
    for _, v in detail_header_items.items():
        image = cv2.imread(os.path.join(folderPath, v["file"]))
        overlaylist.append(image)

    i = list(detail_header_items.keys()).index("paint_black")
    header = overlaylist[i]
    draw_color = detail_header_items["paint_black"]["color"]


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    image_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            image = cv2.flip(image, 1)

            key = cv2.waitKey(10)
            if key == ord(" "):
                is_predict = not is_predict
                if is_predict:
                    image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
                    ys, xs = np.nonzero(image_gray)
                    min_y = np.min(ys)
                    max_y = np.max(ys)
                    min_x = np.min(xs)
                    max_x = np.max(xs)
                    image_gray = image_gray[min_y : max_y, min_x : max_x]
                    _, image_gray = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY)
                    image_gray = cv2.resize(image_gray, (28,28)).astype(np.float32) / 255.
                    image_gray = np.array(image_gray, dtype=np.float32)[None, None, :, :]
                    image_gray = torch.from_numpy(image_gray).to(device)
                    with torch.no_grad():
                        logits = model(image_gray)
                        prob = torch.nn.Softmax(dim=1)(logits)
                        pred = torch.argmax(logits[0])
                elif not is_predict:
                    image_canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                

            if not is_predict:
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                        index_finger_tip = hand_landmarks.landmark[8]
                        middle_finger_tip = hand_landmarks.landmark[12]

                        index_finger_tip_x = index_finger_tip.x * image.shape[1]
                        index_finger_tip_y = index_finger_tip.y * image.shape[0]
                        middle_finger_tip_x = middle_finger_tip.x * image.shape[1]
                        middle_finger_tip_y = middle_finger_tip.y * image.shape[0]
                        
                        finger_up = check_finger_up(hand_landmarks)
                        #Select Mode
                        if finger_up[1] == 1 and finger_up[2] == 1:
                            x1, y1 = index_finger_tip_x, index_finger_tip_y
                            distance_of_2_fingers = math.sqrt((index_finger_tip_x - middle_finger_tip_x)**2 + (index_finger_tip_y - middle_finger_tip_y)**2)
                            if distance_of_2_fingers < 100:
                                for key in detail_header_items.keys():
                                    xmin, ymin, xmax, ymax = get_pos(detail_header_items[key]["pos"])
                                    if index_finger_tip_x > xmin and index_finger_tip_x < xmax and index_finger_tip_y > ymin and index_finger_tip_y < ymax:
                                        i = list(detail_header_items.keys()).index(key)
                                        draw_color = detail_header_items[key]["color"]
                                        header = overlaylist[i]

                        #Drawing Mode
                        elif finger_up[1] == 1 and finger_up[2] == 0:
                            if x1 == 0 and y1 == 0:
                                x1, y1 = index_finger_tip_x, index_finger_tip_y

                            if draw_color == (0, 0, 0):
                                cv2.circle(image, (int(index_finger_tip_x), int(index_finger_tip_y)), ERASER_THICKNESS, draw_color, cv2.FILLED)
                                cv2.line(image, (int(x1), int(y1)), (int(index_finger_tip_x), int(index_finger_tip_y)), draw_color, ERASER_THICKNESS)
                                cv2.line(image_canvas, (int(x1), int(y1)), (int(index_finger_tip_x), int(index_finger_tip_y)), draw_color, ERASER_THICKNESS)
                            else:
                                cv2.circle(image, (int(index_finger_tip_x), int(index_finger_tip_y)), BRUSH_THICKNESS, draw_color, cv2.FILLED)
                                cv2.line(image, (int(x1), int(y1)), (int(index_finger_tip_x), int(index_finger_tip_y)), draw_color, BRUSH_THICKNESS)
                                cv2.line(image_canvas, (int(x1), int(y1)), (int(index_finger_tip_x), int(index_finger_tip_y)), draw_color, BRUSH_THICKNESS)

                            x1, y1 = index_finger_tip_x, index_finger_tip_y
            
            elif is_predict:
                cv2.putText(image, f"You are drawing", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                x_axes = 400
                y_axes = 650
                image[y_axes:y_axes+60, x_axes:x_axes+60] = get_overlay(image[y_axes:y_axes+60, x_axes:x_axes+60], get_images(args.image_path, CLASSES[pred]))
                    

                    

            # Chuyển đổi canvas sang ảnh xám
            image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
            # Tạo mặt nạ nhị phân ngược từ ảnh xám (ngưỡng 1 để chỉ phát hiện vùng không phải màu đen)
            _, image_mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY_INV)

            # Chuyển đổi mặt nạ từ xám sang BGR để có thể áp dụng với ảnh màu
            image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
            # Xóa vùng vẽ trên ảnh gốc bằng phép AND với mặt nạ
            image = cv2.bitwise_and(image, image_mask)
            # Kết hợp ảnh gốc đã xóa vùng vẽ với canvas để hiển thị nét vẽ
            image = cv2.bitwise_or(image, image_canvas)
            
            # Đặt hình ảnh header lên phần trên cùng của ảnh
            image[0:125, 0:1280] = header # Đặt header ở trên cùng
            


            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()

if __name__ == "__main__":
    args = get_args()
    main(args)
