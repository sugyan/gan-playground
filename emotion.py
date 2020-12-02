import argparse
import glob
import os
import pandas as pd
import paz.processors as pr
from paz.backend.image import load_image
from paz.applications import HaarCascadeFrontalFace, MiniXceptionFER


class EmotionDetector(pr.Processor):
    def __init__(self):
        super(EmotionDetector, self).__init__()
        self.detect = HaarCascadeFrontalFace(draw=False)
        self.crop = pr.CropBoxes2D()
        self.classify = MiniXceptionFER()

    def call(self, image):
        boxes2D = self.detect(image)["boxes2D"]
        cropped_images = self.crop(image, boxes2D)
        results = []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            results.append(self.classify(cropped_image)["scores"])
        return results


def run(target_dir):
    detect = EmotionDetector()
    indexes = []
    columns = []
    for i, img_file in enumerate(glob.glob(os.path.join(target_dir, "*.png"))):
        image = load_image(img_file)
        predictions = detect(image)
        if len(predictions) != 1:
            continue

        print(f"{i:05d} {img_file}", predictions[0][0].tolist())
        indexes.append(os.path.abspath(img_file))
        columns.append(predictions[0][0])

    df = pd.DataFrame(
        columns,
        index=indexes,
        columns=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    )
    print(df)
    df.to_hdf("emotion.h5", key="df")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()

    run(args.target_dir)
