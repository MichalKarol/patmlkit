from datetime import datetime
from functools import reduce
import cv2 as cv
import math
import numpy as np
import numpy.typing as npt
import operator
import pyclipper


from collections import defaultdict
from itertools import count, product
from itertools import permutations
from os import makedirs, path as pp
from patmlkit.image import read_rgb_image, write_rgb_image
from patmlkit.json import read_json_file, write_json_file
from tqdm import tqdm, trange
from typing import Any, Callable, Dict, List, Literal, Set, Union, Tuple


class COCOLicense:
    def __init__(
        self,
        id: int,
        name: str,
        url: str,
        custom_attributes: Dict[str, Any] | None = None,
    ):
        self.id = id
        self.name = name
        self.url = url
        self._custom_attributes = custom_attributes or dict()

    @property
    def custom_attributes(self) -> Dict[str, Any]:
        return self._custom_attributes

    @custom_attributes.setter
    def custom_attributes(self, custom_attributes: Dict[str, Any]):
        self._custom_attributes = custom_attributes

    def to_dict(self):
        return {
            **self._custom_attributes,
            "id": self.id,
            "name": self.name,
            "url": self.url,
        }

    @staticmethod
    def from_dict(json_data: Dict[str, Any]):
        required_keys = ["id", "name", "url"]
        return COCOLicense(
            *[json_data[key] for key in required_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys
            },
        )


class COCOCategory:
    def __init__(
        self,
        id: int,
        name: str,
        supercategory: str,
        custom_attributes: Dict[str, Any] | None = None,
    ):
        self.id = id
        self.name = name
        self.supercategory = supercategory
        self._custom_attributes = custom_attributes or dict()

    @property
    def custom_attributes(self) -> Dict[str, Any]:
        return self._custom_attributes

    @custom_attributes.setter
    def custom_attributes(self, custom_attributes: Dict[str, Any]):
        self._custom_attributes = custom_attributes

    def to_dict(self):
        return {
            **self._custom_attributes,
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }

    @staticmethod
    def from_dict(json_data: Dict[str, Any]):
        required_keys = ["id", "name", "supercategory"]
        return COCOCategory(
            *[json_data[key] for key in required_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys
            },
        )


class COCOAnnotation:
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        custom_attributes: Dict[str, Any] | None = None,
    ):
        self.ctx = ctx
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self._custom_attributes = custom_attributes or dict()

    @property
    def image(self):
        return self.ctx.images[self.image_id]

    @property
    def category(self):
        return self.ctx.categories[self.category_id]

    @property
    def custom_attributes(self) -> Dict[str, Any]:
        return self._custom_attributes

    @custom_attributes.setter
    def custom_attributes(self, custom_attributes: Dict[str, Any]):
        self._custom_attributes = custom_attributes

    def to_dict(self):
        return {
            **self._custom_attributes,
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "iscrowd": 0,
        }

    @staticmethod
    def from_dict(ctx: "COCOContext", json_data: Dict[str, Any]):
        required_keys = ["id", "image_id", "category_id"]
        return COCOAnnotation(
            ctx,
            *[json_data[key] for key in required_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys
            },
        )


class COCOPointAnnotation(COCOAnnotation):
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        point: List[int],
        custom_attributes: Dict[str, Any] | None = None,
    ):
        COCOAnnotation.__init__(self, ctx, id, image_id, category_id, custom_attributes)
        self.point = point

    def to_dict(self):
        return {**super().to_dict(), "point": self.point}

    @staticmethod
    def from_dict(ctx: "COCOContext", json_data: Dict[str, Any]):
        required_keys = ["id", "image_id", "category_id", "point"]
        return COCOPointAnnotation(
            ctx,
            *[json_data[key] for key in required_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys
            },
        )


BBOxType = Union[Literal["xywh"], Literal["xyxy"], Literal["ccwh"]]


class COCOBBoxAnnotation(COCOAnnotation):
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        bbox: List[int],
        bbox_type: BBOxType = "xywh",
        custom_attributes: Dict[str, Any] | None = None,
    ):
        COCOAnnotation.__init__(self, ctx, id, image_id, category_id, custom_attributes)
        assert len(bbox) == 4, "Bounding box does not have 4 elements"
        self.bbox = bbox
        if bbox_type == "xyxy":
            min_x, min_y, max_x, max_y = bbox
            self.bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
        if bbox_type == "ccwh":
            center_x, center_y, width, height = bbox
            self.bbox = [
                center_x - width // 2,
                center_y - height // 2,
                center_x - width // 2 + width,
                center_y - height // 2 + height,
            ]

    def to_dict(self):
        [width, height] = self.bbox[2:4]
        return {
            **super().to_dict(),
            "area": width * height,
            "bbox": self.bbox,
            "type": "xywh",
        }

    @staticmethod
    def from_dict(ctx: "COCOContext", json_data: Dict[str, Any]):
        required_keys = ["id", "image_id", "category_id", "bbox"]
        optional_keys = ["type"]
        return COCOBBoxAnnotation(
            ctx,
            *[json_data[key] for key in required_keys],
            *[json_data.get(key) for key in optional_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys and key not in optional_keys
            },
        )


class COCOMaskAnnotation(COCOBBoxAnnotation):
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        bbox: List[int],
        segmentation: List[int],
        area: int,
        bbox_type: BBOxType = "xywh",
        custom_attributes: Dict[str, Any] | None = None,
    ):
        COCOBBoxAnnotation.__init__(
            self, ctx, id, image_id, category_id, bbox, bbox_type, custom_attributes
        )
        assert len(segmentation) % 2 == 0, "Segmentation has odd number of elements"
        assert area > 0, "Area is below zero"
        self.segmentation = segmentation
        self.area = area

    def to_dict(self):
        return {
            **super().to_dict(),
            "segmentation": self.segmentation,
            "area": self.area,
        }

    @staticmethod
    def from_dict(ctx: "COCOContext", json_data: Dict[str, Any]):
        required_keys = [
            "id",
            "image_id",
            "category_id",
            "bbox",
            "segmentation",
            "area",
        ]
        optional_keys = ["type"]
        return COCOMaskAnnotation(
            ctx,
            *[json_data[key] for key in required_keys],
            *[json_data.get(key) for key in optional_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys and key not in optional_keys
            },
        )


class COCOImage:
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        width: int,
        height: int,
        file_name: str,
        license_id: int | None = None,
        custom_attributes: Dict[str, Any] | None = None,
    ):
        self.ctx = ctx
        self.id = id
        self.width = width
        self.height = height
        self.file_name = file_name
        self.license_id = license_id
        self.flickr_url = file_name
        self.coco_url = file_name
        self._custom_attributes = custom_attributes or dict()

    @property
    def annotations(self):
        return [
            self.ctx.annotations[ann_id]
            for ann_id in self.ctx.image_annotations.get(self.id, list())
        ]

    @property
    def license(self):
        if not self.license_id:
            return None
        return self.ctx.licenses.get(self.license_id, None)

    @property
    def custom_attributes(self) -> Dict[str, Any]:
        return self._custom_attributes

    @custom_attributes.setter
    def custom_attributes(self, custom_attributes: Dict[str, Any]):
        self._custom_attributes = custom_attributes

    def to_dict(self):
        return {
            **self._custom_attributes,
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.file_name,
            "license": self.license_id,
            "flickr_url": self.file_name,
            "coco_url": self.file_name,
            "date_captured": "",
        }

    @staticmethod
    def from_dict(ctx: "COCOContext", json_data: Dict[str, Any]):
        required_keys = ["id", "width", "height", "file_name"]
        optional_keys = ["license"]
        return COCOImage(
            ctx,
            *[json_data[key] for key in required_keys],
            *[json_data.get(key) for key in optional_keys],
            custom_attributes={
                key: value
                for key, value in json_data.items()
                if key not in required_keys and key not in optional_keys
            },
        )


class COCOContext:
    base_path: str
    licenses: Dict[int, COCOLicense]
    images: Dict[int, COCOImage]
    annotations: Dict[int, COCOAnnotation]
    categories: Dict[int, COCOCategory]
    image_annotations: Dict[int, List[int]]


class COCOStratifiedIterator:
    def __init__(
        self,
        coco_context: COCOContext,
        split: Tuple[float, float],
        folds: List[List[int]],
    ):
        self.coco_context = coco_context
        self.split = split
        self.folds = folds
        fold_part_size = 100 / len(folds)
        train_size, test_size = self.split
        train_folds_len = int(train_size / fold_part_size)
        test_folds_len = int(test_size / fold_part_size)

        base_set = [False] * (train_folds_len - 1) + [True]
        self.permutations = iter(set(permutations(base_set)))
        self.test_coco = self.__create_coco_out_of_folds__(
            [image_id for fold in folds[:test_folds_len] for image_id in fold],
        )

    def to_json(self, json_file_path: str):
        write_json_file(
            json_file_path,
            {
                "split": self.split,
                "folds": self.folds,
            },
        )

    @staticmethod
    def from_json(
        coco_context: COCOContext, json_file_path: str
    ) -> "COCOStratifiedIterator":
        iterator_data = read_json_file(json_file_path)
        assert iterator_data[
            "split"
        ], "COCO Stratified Iterator file does not have split"
        assert iterator_data[
            "folds"
        ], "COCO Stratified Iterator file does not have folds"
        return COCOStratifiedIterator(
            coco_context, iterator_data["split"], iterator_data["folds"]
        )

    def __create_coco_out_of_folds__(self, folds: List[int]) -> "COCO":
        new_coco_context = COCOContext()
        new_coco_context.licenses = self.coco_context.licenses
        new_coco_context.categories = self.coco_context.categories
        new_coco_context.images = {
            key: value
            for key, value in self.coco_context.images.items()
            if key in folds
        }
        new_coco_context.image_annotations = {
            image.id: [ann.id for ann in image.annotations]
            for image in new_coco_context.images.values()
        }

        new_coco_context.annotations = {
            annotation_id: self.coco_context.annotations[annotation_id]
            for annotation_ids in new_coco_context.image_annotations.values()
            for annotation_id in annotation_ids
        }
        new_coco_context.base_path = self.coco_context.base_path
        return COCO(new_coco_context)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple["COCO", "COCO", "COCO"]:
        next_permutation = next(self.permutations)
        train_coco = self.__create_coco_out_of_folds__(
            [
                image_id
                for idx, is_fold_validation in enumerate(next_permutation)
                if not is_fold_validation
                for image_id in self.folds[idx]
            ]
        )
        validation_coco = self.__create_coco_out_of_folds__(
            [
                image_id
                for idx, is_fold_validation in enumerate(next_permutation)
                if is_fold_validation
                for image_id in self.folds[idx]
            ]
        )
        return (train_coco, validation_coco, self.test_coco)


class COCO:
    def __init__(self, coco_context: COCOContext):
        self.coco_context = coco_context
        self._custom_attributes = dict()
        self._custom_info = dict()

    def to_json(self, json_file_path: str):
        write_json_file(json_file_path, self.to_dict())

    @staticmethod
    def from_json(json_file_path: str):
        coco_data = read_json_file(json_file_path)
        return COCO.from_dict(coco_data, json_file_path)

    @property
    def images(self):
        return list(self.coco_context.images.values())

    @property
    def annotations(self):
        return list(self.coco_context.annotations.values())

    @property
    def categories(self):
        return list(self.coco_context.categories.values())

    @property
    def licenses(self):
        return list(self.coco_context.licenses.values())

    def to_dict(self):
        return {
            **self._custom_attributes,
            "images": [image.to_dict() for image in self.images],
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "licenses": [license.to_dict() for license in self.licenses],
            "categories": [category.to_dict() for category in self.categories],
            "info": {
                **self._custom_info,
                "year": datetime.today().year,
                "version": "1",
                "description": "Generated by patmlkit",
                "contributor": "",
                "url": "https://pypi.org/project/patmlkit/",
                "date_created": datetime.today().isoformat(),
            },
        }

    @staticmethod
    def from_dict(coco_data: Dict[str, Any], base_path: str):
        assert isinstance(
            coco_data.get("images"), list
        ), "COCO file does not have section images"
        assert isinstance(
            coco_data.get("categories"), list
        ), "COCO file does not have section categories"
        assert isinstance(
            coco_data.get("annotations"), list
        ), "COCO file does not have section annotations"

        coco_context = COCOContext()

        coco_context.images = {
            image["id"]: COCOImage.from_dict(coco_context, image)
            for image in coco_data.get("images", [])
        }

        coco_context.licenses = {
            license["id"]: COCOLicense.from_dict(license)
            for license in coco_data.get("licenses", [])
        }

        coco_context.categories = {
            category["id"]: COCOCategory.from_dict(category)
            for category in coco_data.get("categories", [])
        }

        coco_context.annotations = {}
        coco_context.image_annotations = {}

        for annotation in coco_data.get("annotations", []):
            if "point" in annotation:
                new_annotation = COCOPointAnnotation.from_dict(coco_context, annotation)
            elif "segmentation" in annotation:
                new_annotation = COCOMaskAnnotation.from_dict(coco_context, annotation)
            elif "bbox" in annotation:
                new_annotation = COCOBBoxAnnotation.from_dict(coco_context, annotation)
            else:
                new_annotation = COCOAnnotation.from_dict(coco_context, annotation)
            coco_context.annotations[new_annotation.id] = new_annotation
            if coco_context.image_annotations.get(new_annotation.image_id) is None:
                coco_context.image_annotations[new_annotation.image_id] = []
            coco_context.image_annotations[new_annotation.image_id].append(
                new_annotation.id
            )

        coco_context.base_path = pp.dirname(base_path)
        return COCO(coco_context)

    @property
    def custom_attributes(self) -> Dict[str, Any]:
        return self._custom_attributes

    @custom_attributes.setter
    def custom_attributes(self, custom_attributes: Dict[str, Any]):
        self._custom_attributes = custom_attributes

    @property
    def custom_info(self) -> Dict[str, Any]:
        return self._custom_info

    @custom_info.setter
    def custom_info(self, info: Dict[str, Any]):
        self._custom_info = info

    def split_into_tiles(
        self,
        tile_size: int,
        overlap: float,
        tiled_dataset_directory: str,
        skip_empty: bool = True,
        skip_below_area: int = 20,
    ) -> "COCO":
        assert tile_size > 0, "Tile size must be positive"
        assert overlap >= 0, "Overlap must be positive or zero"
        assert overlap < 1, "Overlap must be less than one"

        makedirs(tiled_dataset_directory, exist_ok=True)
        new_coco_context = COCOContext()
        new_coco_context.licenses = self.coco_context.licenses
        new_coco_context.categories = self.coco_context.categories
        new_coco_context.base_path = tiled_dataset_directory
        new_coco_context.images = {}
        new_coco_context.annotations = {}
        new_coco_context.image_annotations = {}

        tile_window_size = int(tile_size * (1 - overlap))

        image_id_generator = count()
        annotation_id_generator = count()
        split_annotations_map = defaultdict(list)

        def adjust_points(points: List[Tuple[int, int]], row, column):
            def isPointInside(point):
                return (
                    row * tile_window_size
                    <= point[0]
                    <= row * tile_window_size + tile_size
                    and column * tile_window_size
                    <= point[1]
                    <= column * tile_window_size + tile_size
                )

            if all([isPointInside(point) for point in points]):
                return [
                    (
                        int(x - row * tile_window_size),
                        int(y - column * tile_window_size),
                    )
                    for (x, y) in points
                ]

            def order_points(pts):
                center = tuple(
                    map(
                        operator.truediv,
                        reduce(lambda x, y: map(operator.add, x, y), pts),
                        [len(pts)] * 2,
                    )
                )
                return sorted(
                    pts,
                    key=lambda coord: (
                        -135
                        - math.degrees(
                            math.atan2(*tuple(map(operator.sub, coord, center))[::-1])
                        )
                    )
                    % 360,
                )

            sorted_points = order_points(np.array(list(set(points))))
            clipPolygon = [
                (row * tile_window_size, column * tile_window_size),
                (row * tile_window_size + tile_size, column * tile_window_size),
                (
                    row * tile_window_size + tile_size,
                    column * tile_window_size + tile_size,
                ),
                (row * tile_window_size, column * tile_window_size + tile_size),
            ]
            SCALING_FACTOR = 10000

            pc = pyclipper.Pyclipper()
            pc.AddPath(
                pyclipper.scale_to_clipper(clipPolygon, SCALING_FACTOR),
                pyclipper.PT_CLIP,
                True,
            )
            pc.AddPath(
                pyclipper.scale_to_clipper(sorted_points, SCALING_FACTOR),
                pyclipper.PT_SUBJECT,
                True,
            )

            solution = pyclipper.scale_from_clipper(
                pc.Execute(pyclipper.CT_INTERSECTION), SCALING_FACTOR
            )

            return [
                (int(x - row * tile_window_size), int(y - column * tile_window_size))
                for (x, y) in order_points(solution[0])
            ]

        def get_annotation_tiles(points: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
            image_tiles = set()
            for x, y in points:
                start_row = max(0, math.ceil((x - tile_size) / tile_window_size))
                end_row = math.ceil(x / tile_window_size)
                start_column = max(0, math.ceil((y - tile_size) / tile_window_size))
                end_column = math.ceil(y / tile_window_size)
                for row, column in product(
                    range(start_row, end_row),
                    range(start_column, end_column),
                ):
                    if (
                        x == row * tile_window_size
                        or y == column * tile_window_size
                        or x == row * tile_window_size + tile_size
                        or y == column * tile_window_size + tile_size
                    ):
                        continue
                    image_tiles.add((row, column))
            return image_tiles

        for annotation in tqdm(
            self.coco_context.annotations.values(), unit="annotation"
        ):
            if isinstance(annotation, COCOPointAnnotation):
                for row, column in get_annotation_tiles([annotation.point]):
                    split_annotations_map[
                        f"{annotation.image_id}_{row}_{column}"
                    ].append(
                        COCOPointAnnotation(
                            new_coco_context,
                            next(annotation_id_generator),
                            -1,
                            annotation.category_id,
                            list(adjust_points([annotation.point], row, column)[0]),
                        )
                    )
            elif isinstance(annotation, COCOMaskAnnotation):
                if len(annotation.segmentation) < 4:
                    continue

                point_list = [
                    (annotation.segmentation[i * 2], annotation.segmentation[i * 2 + 1])
                    for i in range(len(annotation.segmentation) // 2)
                ]

                for row, column in get_annotation_tiles(point_list):
                    new_segmentation_points = adjust_points(point_list, row, column)
                    new_segmentation_list = [
                        coord
                        for point in new_segmentation_points
                        for coord in [point[0], point[1]]
                    ]
                    area = cv.contourArea(
                        np.array([[pt] for pt in new_segmentation_points]).astype(
                            np.float32
                        )
                    )
                    new_x = min((point[0] for point in new_segmentation_points))
                    new_y = min((point[1] for point in new_segmentation_points))
                    new_x_max = max((point[0] for point in new_segmentation_points))
                    new_y_max = max((point[1] for point in new_segmentation_points))
                    new_w = new_x_max - new_x
                    new_h = new_y_max - new_y

                    if new_w == 0 or new_h == 0 or area < skip_below_area:
                        continue

                    split_annotations_map[
                        f"{annotation.image_id}_{row}_{column}"
                    ].append(
                        COCOMaskAnnotation(
                            new_coco_context,
                            next(annotation_id_generator),
                            -1,
                            annotation.category_id,
                            [new_x, new_y, new_w, new_h],
                            "xywh",
                            new_segmentation_list,
                            area,
                        )
                    )
            elif isinstance(annotation, COCOBBoxAnnotation):
                [x, y, w, h] = annotation.bbox
                point_list = [(x, y), (x, y + h), (x + w, y + h), (x + w, y)]

                for row, column in get_annotation_tiles(point_list):
                    new_bbox_points = adjust_points(point_list, row, column)
                    new_x = min((point[0] for point in new_bbox_points))
                    new_y = min((point[1] for point in new_bbox_points))
                    new_x_max = max((point[0] for point in new_bbox_points))
                    new_y_max = max((point[1] for point in new_bbox_points))
                    new_w = new_x_max - new_x
                    new_h = new_y_max - new_y

                    if new_w == 0 or new_h == 0 or new_w * new_h < skip_below_area:
                        continue

                    split_annotations_map[
                        f"{annotation.image_id}_{row}_{column}"
                    ].append(
                        COCOBBoxAnnotation(
                            new_coco_context,
                            next(annotation_id_generator),
                            -1,
                            annotation.category_id,
                            [new_x, new_y, new_w, new_h],
                            "xywh",
                        )
                    )

        for image in tqdm(self.coco_context.images.values(), unit="image"):
            rows = range(0, math.ceil(image.height / tile_window_size))
            columns = range(0, math.ceil(image.width / tile_window_size))
            [base, extension] = pp.splitext(image.file_name)
            read_path = pp.join(self.coco_context.base_path, image.file_name)
            raw_image = read_rgb_image(read_path)

            for row, column in product(rows, columns):
                split_annotations = split_annotations_map[f"{image.id}_{row}_{column}"]
                if skip_empty and len(split_annotations) == 0:
                    continue

                full_tile = np.empty((tile_size, tile_size, 3), np.uint8)
                full_tile.fill(255)

                tile = raw_image[
                    column * tile_window_size : column * tile_window_size + tile_size,
                    row * tile_window_size : row * tile_window_size + tile_size,
                ]
                full_tile[: tile.shape[0], : tile.shape[1]] = tile

                tile_file_name = f"{base}_{row}_{column}{extension}"
                tile_full_path = pp.join(tiled_dataset_directory, tile_file_name)
                write_rgb_image(tile_full_path, full_tile)
                new_image_id = next(image_id_generator)
                new_coco_context.images[new_image_id] = COCOImage(
                    new_coco_context,
                    new_image_id,
                    tile_size,
                    tile_size,
                    tile_file_name,
                    image.license_id,
                )
                new_coco_context.image_annotations[new_image_id] = []
                for split_annotation in split_annotations:
                    split_annotation.image_id = new_image_id
                    new_coco_context.annotations[split_annotation.id] = split_annotation
                    new_coco_context.image_annotations[new_image_id].append(
                        split_annotation.id
                    )

        new_coco = COCO(new_coco_context)
        new_coco.to_json(pp.join(tiled_dataset_directory, "dataset.json"))
        return new_coco

    @staticmethod
    def default_fitness_function(
        chromosome: npt.NDArray[np.int32],
        images: List[COCOImage],
        target_values: Dict[int, int],
        folds: int,
    ):
        fold_categories = defaultdict(lambda: defaultdict(int))
        for image_idx, gene in enumerate(chromosome):
            for ann in images[image_idx].annotations:
                fold_categories[gene][ann.category_id] += 1

        return -1 * np.mean(
            [
                np.sum(
                    np.abs(
                        [
                            category_target_value - fold_categories[i][category_id]
                            for category_id, category_target_value in target_values.items()
                        ]
                    )
                )
                for i in range(folds)
            ]
        ).astype(float)

    def get_stratified_k_fold(
        self,
        split: Tuple[float, float] = (80, 20),
        folds: int = 5,
        population: int = 100,
        max_iterations: int = 100,
        fitness_function: Callable[
            [npt.NDArray[np.int32], List[COCOImage], Dict[int, int], int], float
        ] = default_fitness_function,
        top_population_left_percent: int = 5,
        tournament_size_percent: int = 10,
        mutation_prob: float = 0.01,
    ):
        assert sum(split) == 100, "Split must sum into 100%"
        assert len(split) == 2, "Len of split must equal 2"
        assert all(
            [split % (100 / folds) == 0 for split in split]
        ), "Split must be dividable by 100/folds"
        fold_part_size = 100 / folds
        train_size, _ = split
        train_folds_len = train_size // fold_part_size
        assert train_folds_len > 1, "Train split cannot consists out of single fold"

        rng = np.random.default_rng()

        length_of_chromosome = len(self.images)
        ann_categories = np.array([ann.category_id for ann in self.annotations])
        target_categories_count = {
            category.id: np.sum(ann_categories == category.id) // folds
            for category in self.categories
        }

        number_of_parents = 2

        population_array = (
            rng.random((population, length_of_chromosome)) * folds
        ).astype(np.int32)
        fitness_function_simple = lambda chromosome: fitness_function(
            chromosome, self.images, target_categories_count, folds
        )
        top_population_left = population * top_population_left_percent // 100
        tournament_size = population * tournament_size_percent // 100
        best_chromosome = []

        with trange(max_iterations) as pbar:
            for _ in pbar:
                new_population_array = population_array.copy()

                fitness_scores = np.apply_along_axis(
                    fitness_function_simple, 1, new_population_array
                )

                sorted_indexes = np.argsort(fitness_scores)[::-1]
                tournaments = rng.choice(
                    population,
                    (
                        math.ceil((population - top_population_left) / 2),
                        tournament_size,
                    ),
                )
                crossing_points = (
                    rng.random(math.ceil((population - top_population_left) / 2))
                    * length_of_chromosome
                )

                mutations_mask = (
                    rng.random((population - top_population_left, length_of_chromosome))
                    < mutation_prob
                )
                mutations = (
                    rng.random((population - top_population_left, length_of_chromosome))
                    * folds
                )

                top_population_items_indexes = sorted_indexes[:top_population_left]
                population_array[:top_population_left] = new_population_array[
                    top_population_items_indexes
                ]

                top_tournaments_parents_indexes = np.sort(tournaments, axis=1)[
                    :, :number_of_parents
                ]

                max_crossed_population = population - top_population_left

                population_array[top_population_left::2] = new_population_array[
                    sorted_indexes[top_tournaments_parents_indexes[:, 0]]
                ][: math.ceil(max_crossed_population / 2)]
                population_array[top_population_left + 1 :: 2] = new_population_array[
                    sorted_indexes[top_tournaments_parents_indexes[:, 1]]
                ][: math.floor(max_crossed_population / 2)]

                cross_indicies = np.array(
                    [
                        np.arange(length_of_chromosome) < crossing_point
                        for crossing_point in crossing_points
                    ]
                )

                population_array[top_population_left::2][
                    cross_indicies[: math.ceil(max_crossed_population / 2)]
                ] = new_population_array[
                    sorted_indexes[top_tournaments_parents_indexes[:, 1]]
                ][
                    : math.ceil(max_crossed_population / 2)
                ][
                    cross_indicies[: math.ceil(max_crossed_population / 2)]
                ]

                population_array[top_population_left + 1 :: 2][
                    np.logical_not(cross_indicies)[
                        : math.floor(max_crossed_population / 2)
                    ]
                ] = new_population_array[
                    sorted_indexes[top_tournaments_parents_indexes[:, 0]]
                ][
                    : math.floor(max_crossed_population / 2)
                ][
                    np.logical_not(cross_indicies)[
                        : math.floor(max_crossed_population / 2)
                    ]
                ]

                population_array[top_population_left:][mutations_mask] = mutations[
                    mutations_mask
                ]
                best_chromosome = new_population_array[np.argmax(fitness_scores)]
                pbar.set_postfix(best_score=fitness_function_simple(best_chromosome))

        iterator_folds = [
            [
                self.images[image_idx].id
                for image_idx, gene in enumerate(best_chromosome)
                if gene == fold
            ]
            for fold in range(folds)
        ]
        coco_iterator = COCOStratifiedIterator(self.coco_context, split, iterator_folds)
        coco_iterator.to_json(
            pp.join(self.coco_context.base_path, "dataset_split.json")
        )
        return coco_iterator
