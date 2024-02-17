import cv2
import pickle
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from mmpose_utils import process_mmdet_results, frame_iter, concat, convert_instance_to_frame
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    
from mmpose_data import CustomVideoDataset
from mmpose_inference import init_pose_model, init_test_pipeline, run_pose_inference, run_pose_tracking
from mmcv.parallel import collate
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from torch.utils.data import DataLoader
from mmpose.apis import vis_pose_tracking_result
from mmpose.datasets import DatasetInfo

from typing import Optional, Sequence, Union

# %%
def get_dataset_info():
    
    import configs._base_.datasets.coco as coco
    import configs._base_.datasets.coco_wholebody as coco_wholebody
    
    dataset_info = {
        "TopDownCocoDataset": coco.dataset_info,
        "TopDownCocoWholeBodyDataset": coco_wholebody.dataset_info
    }
    
    return dataset_info

# %%
def detection_inference(model_config, model_ckpt, video_path, bbox_path,
                        device='cuda:0', det_cat_id=1):
    
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    det_model = init_detector(
        model_config, model_ckpt, device=device.lower())

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'

    output = []
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for img in tqdm(frame_iter(cap), total=nFrames):
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)
        output.append(person_results)

    output_file = bbox_path
    pickle.dump(output, open(str(output_file), 'wb'))
    cap.release()

def detection_inference_batch(model_config, model_ckpt, video_path, bbox_path,
                              device='cuda:0', det_cat_id=1, batch_size=10000):
    
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    det_model = init_detector(
        model_config, model_ckpt, device=device.lower())

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Faild to load video file {video_path}'

    output = []
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batched_frames = []
    for img in tqdm(frame_iter(cap), total=nFrames):
        # test a batch of images
        batched_frames.append(img)
        if len(batched_frames) == batch_size:
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results_batched = inference_detector_batched(det_model, batched_frames)
            batched_frames = []
            for mmdet_results in mmdet_results_batched:
                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, det_cat_id)
                output.append(person_results)

    if len(batched_frames) > 0:
        mmdet_results_batched = inference_detector_batched(det_model, batched_frames)
        batched_frames = []
        for mmdet_results in mmdet_results_batched:
                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, det_cat_id)
                output.append(person_results)

    output_file = bbox_path
    pickle.dump(output, open(str(output_file), "wb"))
    cap.release()

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
    
# %%
def pose_inference(model_config, model_ckpt, video_path, bbox_path, pkl_path,
                   video_out_path, device='cuda:0', batch_size=64,
                   bbox_thr=0.95, visualize=True, save_results=True):
    """Run pose inference on custom video dataset"""

    # init model
    model = init_pose_model(model_config, model_ckpt, device)
    model_name = model_config.split("/")[1].split(".")[0]
    print("Initializing {} Model".format(model_name))

    # build data pipeline
    test_pipeline = init_test_pipeline(model)

    # build dataset
    video_basename = video_path.split("/")[-1].split(".")[0]
    dataset = CustomVideoDataset(video_path=video_path,
                                 bbox_path=bbox_path,
                                 bbox_threshold=bbox_thr,
                                 pipeline=test_pipeline,
                                 config=model.cfg)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate)
    print("Building {} Custom Video Dataset".format(video_basename))

    # run pose inference
    print("Running pose inference...")
    instances = []
    for batch in tqdm(dataloader):
        batch['img'] = batch['img'].to(device)
        batch['img_metas'] = [img_metas[0] for img_metas in batch['img_metas'].data]
        with torch.no_grad():
            result = run_pose_inference(model, batch)
        instances.append(result)

    # concat results and transform to per frame format
    results = concat(instances)
    results = convert_instance_to_frame(results, dataset.frame_to_instance)

    # run pose tracking
    results = run_pose_tracking(results)

    # save results
    if save_results:
        print("Saving Pose Results...")
        kpt_save_file = pkl_path
        with open(kpt_save_file, 'wb') as f:
            pickle.dump(results, f)

    # visualzize
    if visualize:
        print("Rendering Visualization...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_save_file = video_out_path
        videoWriter = cv2.VideoWriter(str(video_save_file), fourcc, fps, size)

        dataset = model.cfg.data.test.type
        dataset_info_d = get_dataset_info()
        dataset_info = DatasetInfo(dataset_info_d[dataset])
        for pose_results, img in tqdm(zip(results, frame_iter(cap))):
            for instance in pose_results:
                instance['keypoints'] = instance['preds_with_flip']
            vis_img = vis_pose_tracking_result(model, img, pose_results,
                                               radius=4, thickness=1,
                                               dataset=dataset,
                                               dataset_info=dataset_info,
                                               kpt_score_thr=0.3,
                                               show=False)
            videoWriter.write(vis_img)
        videoWriter.release()
