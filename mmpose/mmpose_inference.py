import copy
import mmcv
import numpy as np
import torch
import torch.nn as nn

from mmpose_utils import LoadImage
from mmcv.runner import load_checkpoint
from mmpose.apis import get_track_id
from mmpose.models import build_posenet
from mmpose.datasets.pipelines import Compose

from mmcv.ops import RoIPool
# from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
# from mmengine.dataset import Compose
from mmdet.utils import get_test_pipeline_cfg
from typing import Optional, Sequence, Union

def init_pose_model(config, checkpoint, device="cuda:0"):
    """Initialize pose model from config file and checkpoint path

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path.
    Returns:
        nn.Module: The constructed detector.
    """
    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None

    model = build_posenet(config.model)
    load_checkpoint(model, checkpoint, map_location=device)
    model.cfg = config
    model.to(device).eval()

    return model


def init_test_pipeline(model):
    """Initialize testing pipeline

    Args:
        model (nn.Module): inference model with config attribute
    Returns:
        pipeline (list[dict | callable]): A sequence of data transforms
    """
    channel_order = model.cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + model.cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    return test_pipeline


def run_pose_inference(model, batch, save_features=False, save_heatmap=False):
    """Defines computations performed for pose inference.

    Args:
        model (nn.Module): inference model with config attribute
        batch (dict): data dictionary with img and img_metas key
        save_feautres (bool): save feature maps
        save_heatmap (bool): save keypoint heatmaps
    Returns:
        result (dict): result dictionary with saved tensors
    """
    img, img_metas = batch['img'], batch['img_metas']
    assert img.size(0) == len(img_metas)
    batch_size, _, img_height, img_width = img.shape

    result = {}
    features = model.backbone(img)
    if model.with_neck:
        features = model.neck(features)
    output_heatmap = model.keypoint_head.inference_model(
                            features, flip_pairs=None)
    keypoint_result = model.keypoint_head.decode(
        img_metas, output_heatmap, img_size=[img_width, img_height])
    if save_features:
        if type(features) is list:
            if type(features[0]) is list:
                features = sum(features, [])
            result['features'] = [x.cpu().numpy() for x in features]
        else:
            result['features'] = features.cpu().numpy()
    if save_heatmap:
        result['output_heatmap'] = output_heatmap
    result['preds'] = keypoint_result['preds']
    result['bbox'] = np.stack([x['image_file'] for x in img_metas])

    img_flipped = img.flip(3)
    features_flipped = model.backbone(img_flipped)
    if model.with_neck:
        features_flipped = model.neck(features_flipped)
    output_flipped_heatmap = model.keypoint_head.inference_model(
        features_flipped, img_metas[0]['flip_pairs'])
    output_heatmap_flipped_avg = (output_heatmap +
                                  output_flipped_heatmap) * 0.5
    keypoint_with_flip_result = model.keypoint_head.decode(
        img_metas, output_heatmap_flipped_avg, img_size=[img_width, img_height])
    if save_features:
        if type(features_flipped) is list:
            if type(features_flipped[0]) is list:
                features_flipped = sum(features_flipped, [])
            result['features_flipped'] = [x.cpu().numpy() for x in features_flipped]
        else:
            result['features_flipped'] = features_flipped.cpu().numpy()
    if save_heatmap:
        result['output_heatmap_flipped_avg'] = output_heatmap_flipped_avg
    result['preds_with_flip'] = keypoint_with_flip_result['preds']

    return result


def run_pose_tracking(results):
    next_id = 0
    pose_result_last = []
    pose_tracked_results = []
    for pose_result in results:
        for instance in pose_result:
            instance['keypoints'] = instance['preds_with_flip']
        pose_result, next_id = get_track_id(pose_result, pose_result_last, next_id,
                                           use_oks=False, tracking_thr=0.3)
        pose_result_last = copy.deepcopy(pose_result)
        for instance in pose_result:
            del instance['keypoints']
        pose_tracked_results.append(pose_result)
    return pose_tracked_results

def inference_detector_batched(
    model: nn.Module,
    imgs: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    data_batched = {}
    data_batched['inputs'] = []
    data_batched['data_samples'] = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_['text'] = text_prompt
            data_['custom_entities'] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_batched['inputs'].append(data_['inputs'])
        data_batched['data_samples'].append(data_['data_samples'])

        # forward the model
    with torch.no_grad():
        result_list = model.test_step(data_batched)

    torch.cuda.empty_cache()

    if not is_batch:
        return result_list[0]
    else:
        return result_list
