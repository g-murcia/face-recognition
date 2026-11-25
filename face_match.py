"""
Face recognition: is this person in the group photo?

Takes a photo of one person and a group photo, finds every face in the
group, and checks each one against the individual using FaceNet-512 embeddings
and cosine distance. Saves an annotated copy of the group photo with a
green box on any match and a red box on everyone else.

Usage:
    python face_match.py --individual images/individual.png --group images/group.png
"""

import argparse
import os

import cv2
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet512"
# RetinaFace is slower than OpenCV's Haar cascades but far more accurate:
# in testing, the Haar detector flagged a hand and a backpack strap as faces.
DETECTOR = "retinaface"
# DeepFace's published cosine-distance threshold for FaceNet-512.
# Lower distance = more similar faces.
DEFAULT_THRESHOLD = 0.30
# Skip weak detections (bags, patterns, background blur).
MIN_FACE_CONFIDENCE = 0.90


def cosine_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_faces(image):
    """Detect, align and embed every face in an image in one pass, so the
    individual and the group faces go through the exact same preprocessing."""
    return DeepFace.represent(
        img_path=image,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual", default="images/individual.png")
    parser.add_argument("--group", default="images/group.png")
    parser.add_argument("--output", default="output/annotated.png")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--show", action="store_true", help="open a preview window")
    args = parser.parse_args()

    individual_img = cv2.imread(args.individual)
    group_img = cv2.imread(args.group)
    if individual_img is None or group_img is None:
        raise SystemExit("Could not read one of the images. Check the paths.")

    individual_faces = embed_faces(individual_img)
    if not individual_faces:
        raise SystemExit("No face found in the individual image.")
    # Use the largest face in the individual photo (the subject).
    target = max(individual_faces,
                 key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])["embedding"]

    group_faces = [
        f for f in embed_faces(group_img)
        if f.get("face_confidence", 1) >= MIN_FACE_CONFIDENCE
    ]
    if not group_faces:
        raise SystemExit("No faces found in the group image.")

    annotated = group_img.copy()
    matches = 0
    print(f"Found {len(group_faces)} face(s) in the group photo.")
    for i, face in enumerate(group_faces, start=1):
        distance = cosine_distance(target, face["embedding"])
        is_match = distance < args.threshold
        matches += is_match
        print(f"  Face {i}: distance {distance:.3f} -> {'MATCH' if is_match else 'no match'}")

        area = face["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        color = (0, 255, 0) if is_match else (0, 0, 255)
        label = f"Match {distance:.2f}" if is_match else f"No match {distance:.2f}"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, label, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print("Result:", "the individual IS in the group photo."
          if matches else "the individual is NOT in the group photo.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to {args.output}")

    if args.show:
        cv2.imshow("Face recognition", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
