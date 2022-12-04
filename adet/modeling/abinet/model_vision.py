import logging
import numpy as np
import torch
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.structures.boxes import Boxes
from detectron2.modeling.postprocessing import detector_postprocess as d2_postprocesss
from adet.utils.comm import get_world_size


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    In addition to the post processing of detectron2, we add scalign for 
    bezier control points.
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = d2_postprocesss(results, output_height, output_width, mask_threshold)

    # scale bezier points
    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)

    return results


def build_top_module(cfg):
    top_type = cfg.MODEL.TOP_MODULE.NAME
    if top_type == "conv":
        inp = cfg.MODEL.FPN.OUT_CHANNELS
        oup = cfg.MODEL.TOP_MODULE.DIM
        top_module = nn.Conv2d(
            inp, oup,
            kernel_size=3, stride=1, padding=1)
    else:
        top_module = None
    return top_module


@META_ARCH_REGISTRY.register()
class ABIVision(GeneralizedRCNN):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_weight       = cfg.MODEL.ABINET.VISION_LOSS_WEIGHT
        self.vision_only       = cfg.MODEL.ABINET.VISION_ONLY
        self.max_num_instances = cfg.MODEL.ABINET.MAX_INS_PER_BATCH // get_world_size()
        self.max_len           = cfg.MODEL.BATEXT.NUM_CHARS
        self.use_aet           = cfg.MODEL.BATEXT.USE_AET
        self.aet_thresh        = cfg.MODEL.BATEXT.AET_THRESH
        self.top_module = build_top_module(cfg)

        if cfg.MODEL.ABINET.VISION_CHECKPOINT is not None:
            logger = logging.getLogger("adet.trainer")
            logger.info(f'Read vision model from {cfg.MODEL.ABINET.VISION_CHECKPOINT}.')
            self.load(cfg.MODEL.ABINET.VISION_CHECKPOINT)
    
    def load(self, source, device=None, strict=True):
        state = torch.load(source, map_location=device)
        self.load_state_dict(state['model'], strict=strict)

    def _compute_index(self, total_num_instances, max_num_instances):
        #assert len(total_num_instances) <= max_num_instances
        total_num_instances = np.array(total_num_instances)
        asc_idx = np.argsort(total_num_instances)
        asc_array = np.sort(total_num_instances)
        buckets = [np.sum(asc_array[:i]) + asc_array[i] * (len(asc_array) - i) for i in range(len(asc_array))]
        reduce_mask = np.array(buckets) > max_num_instances
        reduce_mask = reduce_mask[np.argsort(asc_idx)]
        base_value = int((max_num_instances - total_num_instances[~reduce_mask].sum()) / reduce_mask.sum())
        num_instances = np.where(reduce_mask, base_value, total_num_instances)
        index = [np.random.choice(l, n, replace=False) for l, n in zip(total_num_instances, num_instances)]
        return index

    def preprocess_instance(self, batched_inputs):
        total_num_instances = [len(p['instances']) for p in batched_inputs]
        if sum(total_num_instances) > self.max_num_instances:
            # TODO: prompt
            indexs = self._compute_index(total_num_instances, self.max_num_instances)
            for idx, inputs in zip(indexs, batched_inputs):
                fields = inputs['instances']._fields
                for k, v in fields.items():
                    fields[k] = fields[k][idx]
                inputs['instances']._fields = fields
        for bi in batched_inputs:
            text = bi['instances']._fields['text'][:, :self.max_len]
            bi['instances']._fields['text'] = text
        return batched_inputs

    def preprocess_instance_aet(self, batched_inputs, proposals):
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        gt_beziers = [p.beziers for p in gt_instances]
        gt_targets = [x.text for x in gt_instances]
        gt_boxes = [x.gt_boxes for x in gt_instances]
        gt_classes = [x.gt_classes for x in gt_instances]
        proposals = proposals['proposals']

        gt_num_instances = sum([len(p['instances']) for p in batched_inputs])
        aet_max_instance = self.max_num_instances - gt_num_instances
        proposals_out = self._select_targets(proposals, gt_instances, aet_max_instance)
        proposal_beziers, proposal_targets, proposal_boxes, proposal_classes = proposals_out

        beziers = [torch.cat([gtb, ppb]) for gtb, ppb in zip(gt_beziers, proposal_beziers)]
        targets = [torch.cat([gtt, ppt]) for gtt, ppt in zip(gt_targets, proposal_targets)]
        boxes = [Boxes(torch.cat([gtt.tensor, ppt.tensor])) for gtt, ppt in zip(gt_boxes, proposal_boxes)]
        classes = [torch.cat([gtt, ppt]) for gtt, ppt in zip(gt_classes, proposal_classes)]

        for bi, bezier, target, box, clazz in zip(batched_inputs, beziers, targets, boxes, classes):
            bi['instances']._fields['beziers'] = bezier
            bi['instances']._fields['text'] = target
            bi['instances']._fields['gt_boxes'] = box
            bi['instances']._fields['gt_classes'] = clazz
        return batched_inputs

    def _select_targets(self, proposals, targets, max_instance=None):
        proposal_beziers = []
        proposal_targets = []
        proposal_boxes = []
        proposal_classes = []
        for proposal, target in zip(proposals, targets):
            if proposal.top_feat.size(0) <= 0:
                proposal_beziers.append(proposal.top_feat.new_tensor([]))
                proposal_targets.append(proposal.top_feat.new_tensor([]))
                proposal_boxes.append(Boxes(proposal.top_feat.new_tensor([])))
                proposal_classes.append(proposal.top_feat.new_tensor([]))
                continue
            dis = proposal.top_feat[:, None] - target.beziers[None, :]
            dis = torch.abs(dis).mean(-1)
            text_idx = dis.argmin(-1)
            text = target.text[text_idx][:,:self.max_len]
            gt_boxes = target.gt_boxes[text_idx]
            gt_classes = target.gt_classes[text_idx]

            height, width = proposal.image_size
            thresh = (height * width) ** 0.5 * self.aet_thresh
            valid_idx = dis.min(-1)[0] < thresh
            proposal_beziers.append(proposal.top_feat[valid_idx])
            proposal_targets.append(text[valid_idx])
            proposal_boxes.append(gt_boxes[valid_idx])
            proposal_classes.append(gt_classes[valid_idx])

        total_num_instances = [len(p) for p in proposal_beziers]
        if max_instance is not None and sum(total_num_instances) > max_instance:
            indexs = self._compute_index(total_num_instances, max_instance)
            proposal_beziers = [b[i] for i, b in zip(indexs, proposal_beziers)]
            proposal_targets = [t[i] for i, t in zip(indexs, proposal_targets)]
            proposal_boxes = [t[i] for i, t in zip(indexs, proposal_boxes)]
            proposal_classes = [t[i] for i, t in zip(indexs, proposal_classes)]

        return proposal_beziers, proposal_targets, proposal_boxes, proposal_classes

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator( 
                images, features, gt_instances, self.top_module)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        with torch.no_grad():
            # preprocess instance in vision model and throughout the entire lifecycle
            # this only influences recognition
            batched_inputs = self.preprocess_instance(batched_inputs)
            # use adaptive end-to-end training
            if self.use_aet: 
                batched_inputs = self.preprocess_instance_aet(batched_inputs, proposals)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        _, detector_losses, prediction = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses = {k: self.loss_weight * v for k, v in losses.items()}

        if self.vision_only:
            return losses
        else:
            return losses, prediction

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(
                    images, features, None, self.top_module)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _, prediction = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            results = ABIVision._postprocess(results, batched_inputs, images.image_sizes)

        if self.vision_only:
            return results
        else:
            return results, prediction

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results