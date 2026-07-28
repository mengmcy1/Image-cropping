import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}
INPUT_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description='Run shiye_V1 ONNX inference.')
    parser.add_argument('--onnx', default='models/shiye_V1/model.onnx')
    parser.add_argument('--input', default='inputs/shiye_V1/images')
    parser.add_argument('--output', default='outputs/shiye_V1_onnx')
    parser.add_argument('--threshold', default=0.5, type=float)
    return parser.parse_args()


def list_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    return sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_image(image_path):
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Failed to read image: {image_path}')

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] > 3:
        image = image[:, :, :3]

    return image


def preprocess(image):
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    normalized = (resized - MEAN * 255.0) / (STD * 255.0)
    normalized = normalized / 255.0
    return normalized.transpose(2, 0, 1)[None].astype(np.float32)


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def save_mask(save_path, mask):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(save_path.suffix, mask)
    if not ok:
        raise ValueError(f'Failed to write image: {save_path}')
    encoded.tofile(str(save_path))


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    image_paths = list_images(input_root)
    if not image_paths:
        raise FileNotFoundError(f'No images found: {input_root}')

    for image_path in image_paths:
        image = read_image(image_path)
        model_input = preprocess(image)
        logits = session.run(None, {input_name: model_input})[0]
        probability = sigmoid(logits)[0, 0]

        probability = cv2.resize(
            probability,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = (probability > args.threshold).astype(np.uint8) * 255

        save_path = output_root / '0' / image_path.name
        save_mask(save_path, mask)
        print(f'{image_path} -> {save_path}')

    print(f'Done. Outputs saved to: {output_root}')


if __name__ == '__main__':
    main()
