import shutil
import numpy as np
import os.path as osp
from PIL import Image
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from typing import Any, Dict, Iterable, List, Optional
import torch
import os

DENSEPOSE_MASK_KEY = "dp_masks"
DENSEPOSE_IUV_KEYS_WITHOUT_MASK = ["dp_x", "dp_y", "dp_I", "dp_U", "dp_V"]
DENSEPOSE_CSE_KEYS_WITHOUT_MASK = ["dp_x", "dp_y", "dp_vertex", "ref_model"]
DENSEPOSE_ALL_POSSIBLE_KEYS = set(
    DENSEPOSE_IUV_KEYS_WITHOUT_MASK + DENSEPOSE_CSE_KEYS_WITHOUT_MASK + [DENSEPOSE_MASK_KEY]
)

def _maybe_add_bbox(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    if "bbox" not in ann_dict:
        return
    obj["bbox"] = ann_dict["bbox"]
    # obj["bbox_mode"] = BoxMode.XYWH_ABS
    obj["bbox_mode"] = "XYWH_ABS"


def _maybe_add_segm(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    if "segmentation" not in ann_dict:
        return
    segm = ann_dict["segmentation"]
    if not isinstance(segm, dict):
        # filter out invalid polygons (< 3 points)
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            return
    obj["segmentation"] = segm

def _maybe_add_area(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    if "area" not in ann_dict:
        return
    obj["area"] = ann_dict["area"]

def _maybe_add_keypoints(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    if "keypoints" not in ann_dict:
        return
    keypts = ann_dict["keypoints"]  # list[int]
    for idx, v in enumerate(keypts):
        if idx % 3 != 2:
            # COCO's segmentation coordinates are floating points in [0, H or W],
            # but keypoint coordinates are integers in [0, H-1 or W-1]
            # Therefore we assume the coordinates are "pixel indices" and
            # add 0.5 to convert to floating point coordinates.
            keypts[idx] = v + 0.5
    obj["keypoints"] = keypts


def _maybe_add_densepose(obj: Dict[str, Any], ann_dict: Dict[str, Any]):
    for key in DENSEPOSE_ALL_POSSIBLE_KEYS:
        if key in ann_dict:
            obj[key] = ann_dict[key]


def _combine_images_with_annotations(
    dataset_name: str,
    image_root: str,
    img_datas: Iterable[Dict[str, Any]],
    ann_datas: Iterable[Iterable[Dict[str, Any]]],
):

    ann_keys = ["iscrowd", "category_id"]
    dataset_dicts = []

    for img_dict, ann_dicts in zip(img_datas, ann_datas):
        record = {}
        record["file_name"] = osp.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]
        record["dataset"] = dataset_name
        objs = []
        for ann_dict in ann_dicts:
            assert ann_dict["image_id"] == record["image_id"]
            assert ann_dict.get("ignore", 0) == 0
            obj = {key: ann_dict[key] for key in ann_keys if key in ann_dict}
            _maybe_add_bbox(obj, ann_dict)
            _maybe_add_segm(obj, ann_dict)
            _maybe_add_area(obj, ann_dict)
            _maybe_add_keypoints(obj, ann_dict)
            _maybe_add_densepose(obj, ann_dict)
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def load_coco_json(json_file: str, image_root: str):
    """
    Load COCO annotations from a JSON file

    Args:
        json_file: str
            Path to the file to load annotations from
    Returns:
        Instance of `pycocotools.coco.COCO` that provides access to annotations
        data
    """
    coco_api = COCO(json_file)
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    print("Loaded {} images in COCO format from {}".format(len(imgs), json_file))
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images.
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    dataset_records = _combine_images_with_annotations("coco_densepose", image_root, imgs, anns)
    return dataset_records

# function to conceal IUV arry in to numpy with size of 64*64
def _conceal_arry_to_numpy( conceal_arry, x_arry, y_arry, bbr, hw):
    # using the dimension of 256x256 for using integrate function later
    np_new = np.zeros([1,64,64])
    # mute area
    x_,y_,w,h = bbr[0],bbr[1],bbr[2],bbr[3]
    # picture size
    W,H = hw[1], hw[0]
    # transfer the point arry to tensor
    for x,y,z in zip( x_arry, y_arry, conceal_arry ):
        x = int( (x_ + (x * w)/256) * 64 / W )
        y = int( (y_ + (y * h)/256) * 64 / H )
        np_new[0][y][x] = z
    return torch.from_numpy(np_new)

# function to create new contain dp_{I/U/V}_tensor
# return 4*64*64 size tensor and only three tensor have contain
def _get_tensor_IUV( datas ):
    for data in datas:
        hw = [ data["height"], data["width"] ]
        anns = data["annotations"]
        data['IUV_tensor'] = torch.zeros([4,64,64])
        for ann in anns:
            if "dp_I" not in ann:
                continue
            bbox = ann["bbox"]
            dp_I_tensor = _conceal_arry_to_numpy( ann['dp_I'], ann['dp_x'], ann['dp_y'], bbox, hw) / 15
            dp_U_tensor = _conceal_arry_to_numpy( ann['dp_U'], ann['dp_x'], ann['dp_y'], bbox, hw)
            dp_V_tensor = _conceal_arry_to_numpy( ann['dp_V'], ann['dp_x'], ann['dp_y'], bbox, hw)
            data['IUV_tensor'] = data['IUV_tensor'] + torch.cat( (dp_I_tensor, dp_U_tensor, dp_V_tensor, torch.zeros([1,64,64])), 0 ) 
    return datas


def main():
    # load densepose dataset
    pose_path = "data/inference"
    output_path_image = "data/inference/images"
    output_path_mask = "data/inference/masks"
    output_path_uvmap = "data/inference/uvmaps"
    
    os.makedirs(output_path_image, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)
    os.makedirs(output_path_uvmap, exist_ok=True)
    
    pose_json = osp.join(pose_path, "densepose_minival2014.json")
    dataset_dicts = load_coco_json(pose_json, "")
    dataset_dicts = _get_tensor_IUV( dataset_dicts )
    
    for i, dataset_dict in enumerate(dataset_dicts):
        height = dataset_dict['height']
        width = dataset_dict['width']
        annos = dataset_dict['annotations']
        # choose the largest mask
        # TODO: use real mask area instead of box area
        annos = [anno for anno in annos if 'dp_masks' in anno]
        # def area(anno):
        #     x = anno["bbox"]
        #     return x[3] * x[2]
        def area(anno):
            area = anno["area"]
            return area
        areas = [area(anno) for anno in annos]
        # select the largest
        j = np.argmax(areas)
        anno = annos[j]

        polygons = anno["segmentation"]
        if len(polygons) == 0:
            # COCOAPI does not support empty polygons
            return np.zeros((height, width)).astype(np.bool_)
        rles = mask_util.frPyObjects(polygons, height, width)
        rle = mask_util.merge(rles)
        mask = mask_util.decode(rle).astype(np.bool_)
        im = Image.fromarray(mask)
        # TODO: maybe add erosion to cover the whole area
        file_name = dataset_dict['file_name'].split('_')[-1]
        im.save(osp.join(output_path_mask, file_name))

        # copy COCO images to local dirs
        cocodir = osp.join('/mnt/nas/share/datasets/coco/val2017', file_name) 
        targetdir = osp.join(output_path_image, file_name) 
        shutil.copyfile(cocodir, targetdir)
        
        # store upmap
        # the shape of uvmap is 4*64*64 tensor
        # the uvmap[0] is I [0-15]
        # the uvmap[1/2] is U/V [0-1]
        # the uvmap[3] is zeros tensor
        uvmap = dataset_dict['IUV_tensor']
        torch.save( uvmap, osp.join(output_path_uvmap, file_name) )
        
        # only create 600 images
        # if i >= 600:
        #     break

        # the RLE masks are smoother but not tight 
        # rles = anno["dp_masks"]
        # # filter empty rles
        # rles = [r for r in rles if len(r) > 0]
        # rle = mask_util.merge(rles)
        # mask = mask_util.decode(rle).astype(np.bool_)
        # im = Image.fromarray(mask)
        # im.save('rle_{}_{}.png'.format(i, j))

if __name__ == "__main__":
    main()