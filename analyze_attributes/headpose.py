import argparse
import glob
import math
import pathlib
import cv2
import dlib
import numpy as np
from typing import Dict, List, Optional


class HeadposeDetector:
    def __init__(self) -> None:
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
    def __call__(self, img_file: pathlib.Path) -> List[Optional[float]]:
        image = cv2.imread(str(img_file))
        size = image.shape

        # 2D image points
        points = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = self.detector(rgb, 1)
        if len(dets) != 1:
            return [None, None, None]

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

        return [
            math.degrees(math.asin(math.sin(math.radians(a))))
            for a in euler_angles.flatten()
        ]


def predict(target_dir: pathlib.Path) -> Dict[str, np.ndarray]:
    results = {}
    detect = HeadposeDetector()
    for i, img_file in enumerate(
        map(pathlib.Path, glob.glob(str(target_dir / "*.png")))
    ):
        yaw, pitch, roll = detect(img_file)
        print(f"{i:05d} {img_file}", yaw, pitch, roll)
        results[str(img_file.resolve())] = [yaw, pitch, roll]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=pathlib.Path)
    parser.add_argument(
        "--out_file", type=pathlib.Path, default=pathlib.Path("headposes.npz")
    )
    args = parser.parse_args()

    results = predict(args.target_dir.resolve(strict=True))
    np.savez(args.out_file.resolve(), **results)
