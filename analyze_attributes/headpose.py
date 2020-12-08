import argparse
import glob
import os
import cv2
import dlib
import numpy as np
import pandas as pd


class HeadposeDetector:
    def __init__(self):
        predictor_path = "shape_predictor_68_face_landmarks.dat"

        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0),  # Right mouth corner
            ]
        )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    # https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    def __call__(self, image):
        size = image.shape

        # 2D image points
        points = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = self.detector(rgb, 1)
        if len(dets) != 1:
            return None, None, None

        d = dets[0]
        shape = self.predictor(rgb, d)
        for i in [30, 8, 36, 45, 48, 54]:
            points.append([shape.part(i).x, shape.part(i).y])
        image_points = np.array(points, dtype=np.float64)

        # Camera internals
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )

        # Calculate rotation vector and translation vector
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Calculate euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            cv2.hconcat([rotation_mat, translation_vector])
        )

        (nose_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]),
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs,
        )
        for p in image_points:
            cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(image, p1, p2, (255, 0, 0), 2)

        # Display image
        cv2.imshow("Output", image)
        cv2.waitKey(0)

        return euler_angles.flatten()


def predict(target_dir, data_file):
    detect = HeadposeDetector()
    indexes = []
    columns = []
    for i, img_file in enumerate(glob.glob(os.path.join(target_dir, "*.png"))):
        # if not (img_file.endswith("04338.png") or img_file.endswith("00727.png")):
        # if not (img_file.endswith("04737.png") or img_file.endswith("02743.png")):
        #     continue
        # Read Image
        im = cv2.imread(img_file)
        yaw, pitch, roll = detect(im)
        print(f"{i:05d} {img_file}", yaw, pitch, roll)
        indexes.append(os.path.abspath(img_file))
        columns.append([yaw, pitch, roll])

    df = pd.DataFrame(columns, index=indexes, columns=["pitch", "yaw", "roll"])
    print(df)
    df.to_hdf(data_file, key="df")


def calc_vectors(df, out_file):
    k = round(len(df) / 100.0)
    outputs = {}
    for e in ["pitch", "yaw", "roll"]:
        print(f"top {k} images of {e}")
        print("min:")
        mins = []
        for index, row in df.sort_values(e, ascending=True)[:k].iterrows():
            print(index, row[e])
            mins.append(np.load(f"{index}.npy"))
        print("max:")
        maxs = []
        for index, row in df.sort_values(e, ascending=False)[:k].iterrows():
            print(index, row[e])
            maxs.append(np.load(f"{index}.npy"))
        outputs[e] = np.mean(maxs, axis=0) - np.mean(mins, axis=0)
    np.savez(out_file, **outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    parser.add_argument("--data_file", type=str, default="headposes.h5")
    parser.add_argument("--out_file", type=str, default="headposes.npz")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        predict(args.target_dir, args.data_file)
    df = pd.read_hdf(args.data_file)
    calc_vectors(df, args.out_file)
