import cv2
import numpy as np
import pickle


from mmpose_constants import get_flip_pair_dict
from mmpose_utils import _xyxy2xywh, _box2cs, frame_iter
from torch.utils.data import Dataset


class CustomVideoDataset(Dataset):
    """Create custom video dataset for top down inference

    Args:
        video_path (str): Path to video file
        bbox_path (str): Path to bounding box file
                         (expects format to be xyxy [left, top, right, bottom])
        pipeline (list[dict | callable]): A sequence of data transforms
    """

    def __init__(self,
                 video_path,
                 bbox_path,
                 bbox_threshold,
                 pipeline,
                 config):

        # load video
        self.capture = cv2.VideoCapture(video_path)
        assert self.capture.isOpened(), f'Failed to load video file {video_path}'
        self.frames = np.stack([x for x in frame_iter(self.capture)])

        # load bbox
        self.bboxs = pickle.load(open(bbox_path, "rb"))
        self.bbox_threshold = bbox_threshold

        # create instance to frame and frame to instance mapping
        self.instance_to_frame = []
        self.frame_to_instance = []
        for i, frame in enumerate(self.bboxs):
            self.frame_to_instance.append([])
            for j, instance in enumerate(frame):
                bbox = instance['bbox'] 
                if bbox[4] >= bbox_threshold:
                    self.instance_to_frame.append([i, j])
                    self.frame_to_instance[-1].append(len(self.instance_to_frame) - 1)
    
        self.pipeline = pipeline
        self.cfg = config
        flip_pair_dict = get_flip_pair_dict()
        self.flip_pairs = flip_pair_dict[self.cfg.data.test.type]

    def __len__(self):
        return len(self.instance_to_frame)

    def __getitem__(self, idx):
        frame_num, detection_num = self.instance_to_frame[idx]
        num_joints = self.cfg.data_cfg['num_joints']
        bbox_xyxy = self.bboxs[frame_num][detection_num]['bbox']
        bbox_xywh = _xyxy2xywh(bbox_xyxy)
        center, scale = _box2cs(self.cfg, bbox_xywh)

        # joints_3d and joints_3d_visalble are place holders
        # but bbox in image file, image file is not used but we need bbox information later
        data = {'img': self.frames[frame_num],
                'image_file': bbox_xyxy,
                'center': center,
                'scale': scale,
                'bbox_score': bbox_xywh[4] if len(bbox_xywh) == 5 else 1,
                'bbox_id': 0,
                'joints_3d': np.zeros((num_joints, 3)),
                'joints_3d_visible': np.zeros((num_joints, 3)),
                'rotation': 0,
                'ann_info':{
                    'image_size': np.array(self.cfg.data_cfg['image_size']),
                    'num_joints': num_joints,
                    'flip_pairs': self.flip_pairs
        }}
        data = self.pipeline(data)
        return data
