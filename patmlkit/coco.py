from cgi import print_arguments
import math
from typing import Callable, DefaultDict, Dict, List, Literal, Union, Tuple
from unicodedata import category
from patmlkit.json import read_json_file
from patmlkit.image import read_rgb_image, write_rgb_image
from collections import defaultdict
from os import makedirs, path as pp
from itertools import count, product
import numpy as np
import cv2 as cv
import numpy.typing as npt


class COCOLicense:
    def __init__(self, id: int, name: str, url: str):
        self.id = id
        self.name = name
        self.url = url

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
        }


class COCOCategory:
    def __init__(self, id: int, name: str, supercategory: str):
        self.id = id
        self.name = name
        self.supercategory = supercategory

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }


class COCOAnnotation:
    def __init__(self, ctx: "COCOContext", id: int, image_id: int, category_id: int):
        self.ctx = ctx
        self.id = id
        self.image_id = image_id
        self.category_id = category_id

    @property
    def image(self):
        return self.ctx.images[self.image_id]

    @property
    def category(self):
        return self.ctx.categories[self.category_id]

    def to_dict(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "iscrowd": 0,
        }


class COCOPointAnnotation(COCOAnnotation):
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        point: List[int],
    ):
        COCOAnnotation.__init__(self, ctx, id, image_id, category_id)
        self.point = point

    def to_dict(self):
        return {**super().to_dict(), "point": self.point}


BBOxType = Union[Literal["xywh"], Literal["xyxy"], Literal["ccwh"]]


class COCOBBoxAnnotation(COCOAnnotation):
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        bbox: List[int],
        bbox_type: BBOxType,
    ):
        COCOAnnotation.__init__(self, ctx, id, image_id, category_id)
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


class COCOMaskAnnotation(COCOBBoxAnnotation):
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        image_id: int,
        category_id: int,
        bbox: List[int],
        bbox_type: BBOxType,
        segmentation: List[int],
        area: int,
    ):
        COCOBBoxAnnotation.__init__(
            self, ctx, id, image_id, category_id, bbox, bbox_type
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


class COCOImage:
    def __init__(
        self,
        ctx: "COCOContext",
        id: int,
        width: int,
        height: int,
        file_name: str,
        license_id: int | None = None,
    ):
        self.ctx = ctx
        self.id = id
        self.width = width
        self.height = height
        self.file_name = file_name
        self.license_id = license_id
        self.flickr_url = file_name
        self.coco_url = file_name

    @property
    def annotations(self):
        return self.ctx.image_annotations.get(self.id, list())

    @property
    def license(self):
        if not self.license_id:
            return None
        return self.ctx.licenses.get(self.license_id, None)

    def to_dict(self):
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.file_name,
            "license": self.license_id,
            "flickr_url": self.file_name,
            "coco_url": self.file_name,
            "date_captured": "",
        }


class COCOContext:
    licenses: Dict[int, COCOLicense] = dict()
    images: Dict[int, COCOImage] = dict()
    annotations: Dict[int, COCOAnnotation] = dict()
    categories: Dict[int, COCOCategory] = dict()
    image_annotations: DefaultDict[int, List[COCOAnnotation]] = defaultdict(list)


class COCO:
    def __init__(self, coco_context: COCOContext):
        self.coco_context = coco_context

    @staticmethod
    def from_json_file(json_file_path: str) -> "COCO":
        coco_data = read_json_file(json_file_path)
        assert coco_data["images"], "COCO file does not have section images"
        assert coco_data["categories"], "COCO file does not have section categories"
        assert coco_data["annotations"], "COCO file does not have section annotations"

        coco_context = COCOContext()

        coco_context.images = {
            image["id"]: COCOImage(
                coco_context,
                image["id"],
                image["width"],
                image["height"],
                image["file_name"],
                image["license"],
            )
            for image in coco_data.get("images", [])
        }

        coco_context.licenses = {
            license["id"]: COCOLicense(
                license["id"],
                license["name"],
                license["url"],
            )
            for license in coco_data.get("licenses", [])
        }

        coco_context.categories = {
            category["id"]: COCOCategory(
                category["id"],
                category["name"],
                category["supercategory"],
            )
            for category in coco_data.get("categories", [])
        }

        for annotation in coco_data.get("annotations", []):
            if "point" in annotation:
                new_annotation = COCOPointAnnotation(
                    coco_context,
                    annotation["id"],
                    annotation["image_id"],
                    annotation["category_id"],
                    annotation["point"],
                )
            elif "segmentation" in annotation:
                new_annotation = COCOMaskAnnotation(
                    coco_context,
                    annotation["id"],
                    annotation["image_id"],
                    annotation["category_id"],
                    annotation["bbox"],
                    annotation.get("type", "xywh"),
                    annotation["segmentation"],
                    annotation["area"],
                )
            elif "bbox" in annotation:
                new_annotation = COCOBBoxAnnotation(
                    coco_context,
                    annotation["id"],
                    annotation["image_id"],
                    annotation["category_id"],
                    annotation["bbox"],
                    annotation.get("type", "xywh"),
                )
            else:
                new_annotation = COCOAnnotation(
                    coco_context,
                    annotation["id"],
                    annotation["image_id"],
                    annotation["category_id"],
                )
            coco_context.annotations[new_annotation.id] = new_annotation
            coco_context.image_annotations[new_annotation.image_id].append(
                new_annotation
            )
        return COCO(coco_context)

    @staticmethod
    def from_test() -> "COCO":
        coco_context = COCOContext()

        coco_context.images = {
             1: COCOImage(
                coco_context,
                1,
                100,
                100,
                "xd",
                0,
            ),
             2: COCOImage(
                coco_context,
                2,
                100,
                100,
                "xd2",
                0,
            ),
             3: COCOImage(
                coco_context,
                3,
                100,
                100,
                "xd3",
                0,
            ),

        }

        coco_context.licenses = {
            0: COCOLicense(
                0,
                "xdlic",
                "XD",
            )
        }

        coco_context.categories = {
            0: COCOCategory(
                0,
                "a",
               "False",
            ),
            1: COCOCategory(
                1,
                "a",
               "False",
            )
        }

        coco_context.annotations = {
            0: COCOPointAnnotation(coco_context, 0, 1, 0, [0, 0]),
            1: COCOPointAnnotation(coco_context, 1, 1, 1, [0, 0]),
            2: COCOPointAnnotation(coco_context, 2, 1, 1, [0, 0]),
            3: COCOPointAnnotation(coco_context, 3, 2, 0, [0, 0]),
            4: COCOPointAnnotation(coco_context, 4, 3, 0, [0, 0]),
            5: COCOPointAnnotation(coco_context, 5, 3, 1, [0, 0]),
        }

        coco_context.image_annotations[1].append(coco_context.annotations[0])
        coco_context.image_annotations[1].append(coco_context.annotations[1])
        coco_context.image_annotations[1].append(coco_context.annotations[2])
        coco_context.image_annotations[2].append(coco_context.annotations[3])
        coco_context.image_annotations[2].append(coco_context.annotations[4])
        coco_context.image_annotations[3].append(coco_context.annotations[5])

        return COCO(coco_context)


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
            "images": self.images,
            "annotations": self.annotations,
            "licenses": self.licenses,
            "categories": self.categories,
        }

    def split_into_tiles(self, tile_size: int, overlap: float, tiled_dataset_directory: str, skip_empty: bool = True) -> "COCO":
        makedirs(tiled_dataset_directory, exist_ok=True)
        new_coco_context = COCOContext()
        new_coco_context.licenses = self.coco_context.licenses
        new_coco_context.categories = self.coco_context.categories

        tile_window_size = int(tile_size * (1 - overlap))

        image_id_generator = count()
        annotation_id_generator = count()
        split_annotations_map = defaultdict(list)

        def adjust_points(points: List[Tuple[int, int]], row, column):
            return {(
                max(min(x, row * tile_window_size), row * tile_window_size + tile_size),
                max(min(y, column * tile_window_size), column * tile_window_size + tile_size)
            ) for x, y in points}

        for annotation in self.coco_context.annotations.values():
            if isinstance(annotation, COCOPointAnnotation):
                x, y = annotation.point
                row = math.ceil(x / tile_window_size)
                column = math.ceil(y / tile_window_size)

                split_annotations_map[f"{annotation.image_id}_{row}_{column}"].append(COCOPointAnnotation(
                    new_coco_context,
                    next(annotation_id_generator),
                    -1,
                    annotation.category_id,
                    list(*adjust_points([(x, y)], row, column))
                ))
            elif isinstance(annotation, COCOBBoxAnnotation):
                [x, y, w, h] = annotation.bbox
                point_list = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                image_tiles = set()

                for point in point_list:
                    row = math.ceil(point[0] / tile_window_size)
                    column = math.ceil(point[1] / tile_window_size)
                    image_tiles.add((row, column))

                for row, column in image_tiles:
                    new_bbox_points = adjust_points(point_list, row, column)
                    new_x = min((point[0] for point in new_bbox_points))
                    new_y = min((point[1] for point in new_bbox_points))
                    new_x_max = max((point[0] for point in new_bbox_points))
                    new_y_max = max((point[1] for point in new_bbox_points))
                    new_w = new_x_max - new_x
                    new_h = new_y_max - new_y

                    split_annotations_map[f"{annotation.image_id}_{row}_{column}"].append(COCOBBoxAnnotation(
                        new_coco_context,
                        next(annotation_id_generator),
                        -1,
                        annotation.category_id,
                        [new_x, new_y, new_w, new_h],
                        "xywh"
                    ))
            elif isinstance(annotation, COCOMaskAnnotation):
                [x, y, w, h] = annotation.bbox
                bbox_point_list = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
                point_list = [
                    (annotation.segmentation[i * 2], annotation.segmentation[i * 2 + 1])
                    for i in range(len(annotation.segmentation) // 2)
                ]
                image_tiles = set()

                for point in point_list:
                    row = math.ceil(point[0] / tile_window_size)
                    column = math.ceil(point[1] / tile_window_size)
                    image_tiles.add((row, column))

                for row, column in image_tiles:
                    new_bbox_points = adjust_points(bbox_point_list, row, column)
                    new_x = min((point[0] for point in new_bbox_points))
                    new_y = min((point[1] for point in new_bbox_points))
                    new_x_max = max((point[0] for point in new_bbox_points))
                    new_y_max = max((point[1] for point in new_bbox_points))
                    new_w = new_x_max - new_x
                    new_h = new_y_max - new_y

                    new_segmentation_points = adjust_points(point_list, row, column)
                    new_segmantation_list = [coord for point in new_segmentation_points for coord in [point[0], point[1]] ]

                    split_annotations_map[f"{annotation.image_id}_{row}_{column}"].append(COCOMaskAnnotation(
                        new_coco_context,
                        next(annotation_id_generator),
                        -1,
                        annotation.category_id,
                        [new_x, new_y, new_w, new_h],
                        "xywh",
                        new_segmantation_list,
                        cv.contourArea(new_segmentation_points)
                    ))

            for image in self.coco_context.images.values():
                rows = range(0, math.ceil(image.height / tile_window_size))
                columns = range(0, math.ceil(image.width / tile_window_size))
                [base, extension] = pp.splitext(image.file_name)
                raw_image = read_rgb_image(image.file_name)

                for row, column in product(rows, columns):
                    split_annotations = split_annotations_map[f"{annotation.image_id}_{row}_{column}"]
                    if skip_empty and len(split_annotations) == 0:
                        continue

                    full_tile = np.empty((tile_size, tile_size, 3))
                    full_tile.fill(255)

                    tile = raw_image[
                        row * tile_window_size : row * tile_window_size + tile_size,
                        column * tile_window_size : column * tile_window_size + tile_size,
                    ]
                    full_tile[: tile.shape[0], : tile.shape[1]] = tile

                    tile_file_name = pp.join(tiled_dataset_directory, f"{base}_{row}_{column}.{extension}")
                    write_rgb_image(tile_file_name, full_tile)
                    new_image_id = next(image_id_generator)
                    new_coco_context.images[new_image_id] = COCOImage(new_coco_context, new_image_id, tile_size, tile_size, tile_file_name, image.license_id)
                    for split_annotation in split_annotations:
                        split_annotation.image_id = new_image_id
                        new_coco_context.annotations[split_annotation.id] = split_annotation

        return COCO(new_coco_context)

    @staticmethod
    def default_fitness_function(chromosome: npt.NDArray[np.int32], images: List[COCOImage], target_values: Dict[int, int], folds: int):
        fold_categories = defaultdict(lambda: defaultdict(int))
        for image_idx, gene in enumerate(chromosome):
            for ann in images[image_idx].annotations:
                fold_categories[gene][ann.category_id] += 1

        # print([
        #     np.abs([
        #         category_target_value - fold_categories[i][category_id] for category_id, category_target_value in target_values.items()
        #     ]
        #         ) for i in range(folds)
        #     ])

        return -1 *  np.mean([
            np.sum(np.abs([
                category_target_value - fold_categories[i][category_id] for category_id, category_target_value in target_values.items()
            ]
                )) for i in range(folds)
            ])

    def get_stratified_k_fold(self, splits: List[float] = [80, 20], folds: int = 5, population: int = 100, max_iterations: int = 100, fitness_function: Callable[[npt.NDArray[np.int32], List[COCOImage], Dict[int, int], int], float] = default_fitness_function, top_population_left_percent: int = 10, tournament_size_percent: int = 5, mutation_prob: float = 0.05):
        assert sum(splits) == 100, "Splits must sum into 100%"
        assert all([split % (100 / folds) == 0 for split in splits]), "Splits must be dividable by 100/folds"

        rng = np.random.default_rng()

        # number_of_splits = len(splits)
        length_of_chromosome = len(self.images)
        ann_categories = np.array([ann.category_id for ann in self.annotations])
        target_categories_count = { category.id: np.sum(ann_categories == category.id) // folds for category in self.categories }

        number_of_parents = 2

        population_array = (rng.random((population, length_of_chromosome)) * folds).astype(np.int32)
        fintness_function_simple = lambda chromosome: fitness_function(chromosome, self.images, target_categories_count, folds)
        top_population_left = population * top_population_left_percent // 100
        tournament_size = population * tournament_size_percent // 100

        for iteration in range(max_iterations):
            new_population_array = population_array.copy()
            # print(new_population_array, new_population_array.shape)

            fitness_scores = np.apply_along_axis(fintness_function_simple, 1, new_population_array)
            # print(fitness_scores, fitness_scores.shape)

            sorted_indexes = np.argsort(fitness_scores)[::-1]
            tournaments = rng.choice(population, (population - top_population_left, tournament_size))
            transfer_from_second_parent = rng.random((population - top_population_left, length_of_chromosome)) > 0.5

            mutations_mask = rng.random((population, length_of_chromosome)) < mutation_prob
            mutations = rng.random((population, length_of_chromosome)) * folds

            top_population_items_indexes = sorted_indexes[:top_population_left]
            population_array[:top_population_left] = new_population_array[top_population_items_indexes]

            top_tournaments_parents_indexes = np.sort(tournaments, axis=1)[:, :number_of_parents]

            population_array[top_population_left:] = new_population_array[sorted_indexes[top_tournaments_parents_indexes[:, 0]]]
            # print("pre", "".join([str(i) for i in population_array[top_population_left + idx]]))

            population_array[top_population_left:][transfer_from_second_parent] = new_population_array[sorted_indexes[top_tournaments_parents_indexes[:, 1]]][transfer_from_second_parent]
            # print("post", "".join([str(i) for i in population_array[top_population_left + idx]]))


            # for idx in range(population - top_population_left):
            #     print("pre", "".join([str(i) for i in population_array[top_population_left + idx]]))
            #     print("changed", "".join([str(i) for i in population_array[top_population_left + idx][split_points[idx]:]]))
            #     population_array[top_population_left + idx, split_points[idx]:] = new_population_array[sorted_indexes[top_tournaments_parents_indexes[:, 1]]][idx, split_points[idx]:]
            #

            population_array[mutations_mask] = mutations[mutations_mask]

            best_chromosome = new_population_array[np.argmax(fitness_scores)]
            fold_categories = defaultdict(lambda: defaultdict(int))
            for image_idx, gene in enumerate(best_chromosome):
                for ann in self.images[image_idx].annotations:
                    fold_categories[gene][ann.category_id] += 1

            # print)

            print(iteration, np.max(fitness_scores),
            [
                np.abs([
                    category_target_value - fold_categories[i][category_id] for category_id, category_target_value in target_categories_count.items()
                ]
                    ) for i in range(folds)
                ],

            [
                    category_target_value for category_id, category_target_value in target_categories_count.items()

                ],

                [
                 [
                    fold_categories[i][category_id] for category_id, category_target_value in target_categories_count.items()
                ]
                     for i in range(folds)
                ]

                )

        return None


def main():
    COCO.from_json_file("/home/mkarol/data/dvc_datasets/ki_67/SHIDC_annotated_transformed/train/train.json").get_stratified_k_fold(splits=[100*2/3, 100/3], folds=3, population=100)

if __name__ == "__main__":
    main()
