import cv2
import numpy as np

def frame_iter(capture):
    while capture.grab():
        yield capture.retrieve()[1]


class LoadImage:
    """Simple pipeline step to check channel order"""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, data):
        """Call function to load images into results.
        Args:
            data (dict): A data dict contains the img.
        Returns:
            dict: ``data`` will be returned containing loaded image.
        """
	#data['image_file'] = ''
        if self.color_type == 'color' and self.channel_order == 'rgb':
            img = cv2.cvtColor(data['img'], cv2.COLOR_BGR2RGB)
        data['img'] = img
        return data


def _xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.
    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (4,) or
             (5,). (left, top, right, bottom, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (4,) or (5,). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0] + 1
    bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1] + 1
    return bbox_xywh


def _box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)
    Args:
        x, y, w, h
    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """
    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25
    return center, scale


def concat(instances):
    """Concatenate pose result batches
    Args:
        instances list(dict): list of result dict of each batch
    Returns:
        result dict(array): dict of saved outputs
    """
    assert len(instances) > 0, "Empty instances inputted"

    results = {}
    keys = list(instances[0].keys())

    for k in keys:
        if type(instances[0][k]) is list:
            list_len = len(instances[0][k])
            results[k] = [np.concatenate([instance[k][i] for instance in instances]) for i in range(list_len)]
        else:
            results[k] = np.concatenate([instance[k] for instance in instances])
    
    return results


def convert_instance_to_frame(results, frame_to_instance):
    """Convert pose results from per instance to per frame format
    Args:
        results dict(array): dict of saved outputs
        frame_to_instance list: list of instance idx per frame
    Returns:
        results list(list(dict)): frame list of every instance's result dict
    """
    results_frame = []
    for idxs in frame_to_instance:
        results_frame.append([])
        for idx in idxs:
            result_instance = {k: v[idx] for k, v in results.items()}
            results_frame[-1].append(result_instance)

    return results_frame


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]
    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results
