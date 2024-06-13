import os
import cv2
import json
import argparse
import glob
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def joints_est(file, threshold=0.3):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=threshold) as hands:
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        outputs = {}
        outputs["Left"] = []
        outputs["Right"] = []
        if results.multi_handedness is not None:
            detect_len = len(results.multi_handedness)
            labels = []
            for i in range(detect_len):
                label = results.multi_handedness[i].classification[0].label
                labels.append(label)
            if len(labels) == 2 and labels[0] == labels[1]:
                if results.multi_handedness[0].classification[0].score < results.multi_handedness[1].classification[0].score:
                    if labels[0] == 'Left':
                        labels[0] = 'Right'
                    else:
                        labels[0] = 'Left'
                else:
                    if labels[1] == 'Left':
                        labels[1] = 'Right'
                    else:
                        labels[1] = 'Left'
            for i in range(detect_len):
                label = labels[i]
                hand_landmarks = results.multi_hand_landmarks[i]
                for landmark in hand_landmarks.landmark:
                    outputs[label] = outputs[label] + [1 - landmark.x * 2, landmark.y * 2 - 1, landmark.z * 2 - 1]
    return results, outputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/gqj/hand_sfs/hand3_data')
    parser.add_argument('--if_stream', type=bool, default=False)
    parser.add_argument('--name', type=int, default=2)
    parser.add_argument('--range', type=int, default=None)
    args = parser.parse_args()

    if args.range is not None:
        start = 1
        end = args.range + 1
    else:
        start = args.name
        end = args.name + 1
    for i in range(start, end):
        data_path = "%s/%d/img" % (args.data_path, i)
        print(data_path)
        # For static images:
        IMAGE_FILES = sorted(glob.glob("%s/*.png" % data_path))
        for idx, IMAGE_FILE in enumerate(IMAGE_FILES):
            results, outputs = joints_est(IMAGE_FILE)

            os.makedirs(data_path.replace('img', 'pose'), exist_ok=True)
            with open('%s/%02d.json' % (data_path.replace('img', 'pose'), idx), 'w') as f:
                json.dump(outputs, f, ensure_ascii=False)

        # if results.multi_handedness is not None:
        #     print('Handedness:', results.multi_handedness)
        #
        #     image_height, image_width, _ = image.shape
        #     annotated_image = image.copy()
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(
        #             annotated_image,
        #             hand_landmarks,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing_styles.get_default_hand_landmarks_style(),
        #             mp_drawing_styles.get_default_hand_connections_style())
        #     os.makedirs(data_path.replace('img', 'pose') + '/annotated_image', exist_ok=True)
        #     cv2.imwrite(
        #         data_path.replace('img', 'pose') + '/annotated_image/' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        #
