from typing import DefaultDict, Dict, List, Literal, Union
from patmlkit.json.read_json_file import read_json_file
from collections import defaultdict


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
    def __init__(self, json_path: str):
        coco_data = read_json_file(json_path)
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

        self.coco_context = coco_context

    @property
    def images(self):
        return self.coco_context.images.values()

    @property
    def annotations(self):
        return self.coco_context.annotations.values()

    @property
    def categories(self):
        return self.coco_context.categories.values()

    @property
    def licenses(self):
        return self.coco_context.licenses.values()

    def to_dict(self):
        return {
            "images": self.images,
            "annotations": self.annotations,
            "licenses": self.licenses,
            "categories": self.categories,
        }
