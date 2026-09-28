from pathlib import Path
from PIL import Image


class CUBBoundingBoxes:
    def __init__(self, root):
        self.root = Path(root)

        self.image_paths = {}
        with open(self.root / "images.txt", "r") as f:
            for line in f:
                image_id, path = line.strip().split(maxsplit=1)
                self.image_paths[int(image_id)] = path

        self.boxes = {}
        with open(self.root / "bounding_boxes.txt", "r") as f:
            for line in f:
                parts = line.strip().split()
                image_id = int(parts[0])
                x, y, width, height = map(float, parts[1:5])
                self.boxes[image_id] = (x, y, width, height)

    def get(self, image_id):
        return self.boxes[image_id]

    def get_image_path(self, image_id):
        return self.root / "images" / self.image_paths[image_id]

    def get_transformed_box(self, image_id, size=224):
        """
        Transform the original CUB bounding box using the same geometry as:

            Resize(shortest edge -> size)
            CenterCrop(size, size)

        Returns:
            (x1, y1, x2, y2) clipped to the final image.
        """
        image_path = self.get_image_path(image_id)

        with Image.open(image_path) as image:
            width, height = image.size

        x, y, box_width, box_height = self.get(image_id)

        scale = size / min(width, height)

        resized_width = round(width * scale)
        resized_height = round(height * scale)

        crop_left = max(0, (resized_width - size) // 2)
        crop_top = max(0, (resized_height - size) // 2)

        x1 = x * scale - crop_left
        y1 = y * scale - crop_top
        x2 = (x + box_width) * scale - crop_left
        y2 = (y + box_height) * scale - crop_top

        x1 = max(0.0, min(float(size), x1))
        y1 = max(0.0, min(float(size), y1))
        x2 = max(0.0, min(float(size), x2))
        y2 = max(0.0, min(float(size), y2))

        return x1, y1, x2, y2


if __name__ == "__main__":
    boxes = CUBBoundingBoxes("data/CUB_200_2011")

    image_id = 1

    print("Image:", boxes.get_image_path(image_id))
    print("Original bbox:", boxes.get(image_id))
    print("Transformed bbox:", boxes.get_transformed_box(image_id))
