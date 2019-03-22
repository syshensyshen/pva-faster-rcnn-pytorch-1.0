from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.config import cfg
from models.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import numpy as np

#from models.hyper_rcnn import _hyper_rcnn
from models.features import PyramidFeatures, PyramidFeaturesEx, DilatedPyramidFeaturesEx
from models.pva_faster_rcnn import pva_faster_rcnn
from models.features import resnethyper, resnethyperex
from lib.rpn.proposal_layer import _ProposalLayer_FPN
from lib.rpn.anchor_target_layer import _AnchorTargetLayer_FPN
from lib.rpn.proposal_target_layer import _ProposalTargetLayer
from lib.roi_layers.roi_align_layer import ROIAlignLayer
from lib.roi_layers.roi_pooling_layer import ROIPoolingLayer
from models.smoothl1loss import _smooth_l1_loss

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x) # 1/2
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x) # 1/4

    x = self.layer1(x) # 1/4
    x = self.layer2(x) # 1/8
    x = self.layer3(x) # 1/16
    x = self.layer4(x) # 1/32

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x

class resnet_faster(_fasterRCNN):
  def __init__(self, num_clssses, block, layers, training, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.rpn_din = 512
    self._num_clssses = num_clssses

    _fasterRCNN.__init__(self, num_clssses, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self._num_clssses)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self._num_clssses)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

class resnet_pva(pva_faster_rcnn):
  def __init__(self, num_clssses, num_layers=101, pretrained=False, class_agnostic=False):
      self.dout_base_model = 1024
      self.pretrained = pretrained
      self.class_agnostic = class_agnostic
      self.rcnn_din = 1024
      self.rpn_din = 512
      self.num_layers = num_layers
      pva_faster_rcnn.__init__(self, num_clssses, class_agnostic)

  def _init_modules(self):    
    if self.num_layers==18:
      model = resnethyper(self.n_classes, BasicBlock, [2, 2, 2, 2])
      if self.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='/home/user/.torch/models/'), strict=False)
    elif self.num_layers==34:
      model = resnethyper(self.n_classes, BasicBlock, [3, 4, 6, 3])
      if self.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='/home/user/.torch/models/'), strict=False)
    elif self.num_layers==50:
      model = resnethyper(self.n_classes, Bottleneck, [3, 4, 6, 3])
      if self.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='/home/user/.torch/models/'), strict=False)
    elif self.num_layers==101:
      model = resnethyper(self.n_classes,Bottleneck, [3, 4, 23, 3])
      if self.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='/home/user/.torch/models/'), strict=False)
    elif self.num_layers==152:
      model = resnethyper(self.n_classes, Bottleneck, [3, 8, 36, 3])
      if self.pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='/home/user/.torch/models/'), strict=False)
    else:
      exit()

    self.RCNN_base = model
    
    self.RCNN_cls_score = nn.Linear(self.rcnn_din, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(self.rcnn_din, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(self.rcnn_din, 4 * self.n_classes)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    #print(pool5_flat.shape)
    fc_features = self.RCNN_top(pool5_flat)
    
    return fc_features

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class resnet_fpn(nn.Module):

  @staticmethod
  def reshape(x, d):
      input_shape = x.size()
      x = x.view(
          input_shape[0],
          int(d),
          int(float(input_shape[1] * input_shape[2]) / float(d)),
          input_shape[3]
      )
      return x

  def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        #img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0) / np.log(2)
        roi_level = torch.floor(roi_level + 4)
        # --------
        # roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        # roi_level = torch.round(roi_level + 4)
        # ------
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        
        roi_pool_feats = []
        box_to_levels = []
        for i, l in enumerate(range(2, 6)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero().squeeze()
            #print('sssss', idx_l)
            idx_l = torch.reshape(idx_l, (1,-1))[0]
            box_to_levels.append(idx_l)
            scale = feat_maps[i].size(2) / im_info[0][0]
            feat = self.roi_feat(feat_maps[i], rois[idx_l], scale)
            roi_pool_feats.append(feat)

        roi_pool_feat = torch.cat(roi_pool_feats, 0)
        box_to_level = torch.cat(box_to_levels, 0)
        #idx_sorted, order = torch.sort(box_to_level)
        _, order = torch.sort(box_to_level)
        roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

  def __init__(self, num_clssses, block, layers, training, pretrained=False, class_agnostic=False):
    self.anchor_ratios = cfg.ANCHOR_RATIOS
    self.anchor_scales = cfg.FPN_ANCHOR_SCALES
    self.inplanes = 64
    self.training = training
    self._num_anchors = len(self.anchor_scales[0]) * len(self.anchor_ratios)
    self.din = 256
    self.rpn_din = 512
    self._num_clssses = num_clssses
    self.rcnn_din = 1024
    self.class_agnostic = class_agnostic
    super(resnet_fpn, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    if block == BasicBlock:
        fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
    elif block == Bottleneck:
        fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]     
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    self.freeze_bn()

    self.PyramidFeature = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])

    # rpn regression
    self.rpnfeature = nn.Conv2d(self.din, self.rpn_din, 3, 1, 1, bias=True)
    self.rpn_cls = nn.Conv2d(self.rpn_din, self._num_anchors * 2, 1, 1, 0)
    self.rpn_bbox_pred = nn.Conv2d(self.rpn_din, self._num_anchors * 4, 1, 1, 0)

    if self.training:
      self.anchor_targets = _AnchorTargetLayer_FPN(self.anchor_ratios)
      self.proposal_targets = _ProposalTargetLayer(self._num_clssses)

    # proposal
    self.proposals = _ProposalLayer_FPN(self.anchor_ratios)
    if cfg.POOLING_MODE == 'align':
      self.roi_feat = ROIAlignLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
    else:
      self.roi_feat = ROIPoolingLayer((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)

    # RCNN
    self.rcnnfeatures = nn.Sequential(
      nn.Conv2d(256, self.rcnn_din, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
      nn.ReLU(True),
      nn.Conv2d(self.rcnn_din, self.rcnn_din, kernel_size=1, stride=1, padding=0),
      nn.ReLU(True)
      )

    self.rcnn_cls = nn.Linear(self.rcnn_din, self._num_clssses)
    if self.class_agnostic:
      self.rcnn_bbox_pred = nn.Linear(self.rcnn_din, 4)
    else:
      self.rcnn_bbox_pred = nn.Linear(self.rcnn_din, 4 * self._num_clssses)
    self._init_weights()

  def _init_weights(self):
    normal_init(self.rpnfeature, 0, 0.01)
    normal_init(self.rpn_cls, 0, 0.01)
    normal_init(self.rpn_bbox_pred, 0, 0.001)
    normal_init(self.rcnn_cls, 0, 0.01)
    normal_init(self.rcnn_bbox_pred, 0, 0.001)

  def _make_layer(self, block, planes, blocks, stride=1):
      downsample = None
      if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(planes * block.expansion),
          )

      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))
      self.inplanes = planes * block.expansion
      for i in range(1, blocks):
          layers.append(block(self.inplanes, planes))

      return nn.Sequential(*layers)

  def freeze_bn(self):
      '''Freeze BatchNorm layers.'''
      for layer in self.modules():
          if isinstance(layer, nn.BatchNorm2d):
              layer.eval()

  def forward(self, img_batch, im_info, gt_boxes):

      x = self.conv1(img_batch)
      x = self.bn1(x)
      x = self.relu(x)
      x = self.maxpool(x)

      x1 = self.layer1(x)
      x2 = self.layer2(x1)
      x3 = self.layer3(x2)
      x4 = self.layer4(x3)

      features = self.PyramidFeature([x1, x2, x3, x4])

      rpn_cls_scores = []
      rpn_cls_probs = []
      rpn_bbox_preds = []
      rpn_shapes = []
      for i in range(len(features)):
        feat_map = features[i]
        batch_size = feat_map.size(0)
        
        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.rpnfeature(feat_map), inplace=True)
        # get rpn classification score
        rpn_cls_score = self.rpn_cls(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.sigmoid(rpn_cls_score_reshape)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self._num_anchors * 2)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv1)
        #print(rpn_bbox_pred)

        rpn_shapes.append([rpn_cls_score.size()[2], rpn_cls_score.size()[3]])
        rpn_cls_scores.append(rpn_cls_score.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
        rpn_cls_probs.append(rpn_cls_prob.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))
        rpn_bbox_preds.append(rpn_bbox_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4))
      
      rpn_cls_score_alls = torch.cat(rpn_cls_scores, 1)
      rpn_cls_prob_alls = torch.cat(rpn_cls_probs, 1)
      rpn_bbox_pred_alls = torch.cat(rpn_bbox_preds, 1)
      #print(rpn_bbox_pred_alls)

      if self.training:
        assert gt_boxes is not None
        rois = self.proposals((rpn_cls_prob_alls.data, rpn_bbox_pred_alls.data,
                                 im_info, 'TRAIN', rpn_shapes))

        rpn_data = self.anchor_targets((rpn_cls_score_alls.data, gt_boxes, im_info, rpn_shapes))

        # compute classification loss
        rpn_label = rpn_data[0].view(batch_size, -1)
        rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        rpn_cls_score = torch.index_select(rpn_cls_score_alls.view(-1,2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        rpn_label = Variable(rpn_label.long())
        rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
        #fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

        # compute bbox regression loss
        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights.unsqueeze(2) \
                .expand(batch_size, rpn_bbox_inside_weights.size(1), 4))
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights.unsqueeze(2) \
                .expand(batch_size, rpn_bbox_outside_weights.size(1), 4))
        rpn_bbox_targets = Variable(rpn_bbox_targets)
        #print(rpn_bbox_targets)
        
        rpn_loss_bbox = _smooth_l1_loss(rpn_bbox_pred_alls, rpn_bbox_targets, rpn_bbox_inside_weights, 
                        rpn_bbox_outside_weights, sigma=3)
        
        roi_data = self.proposal_targets(rois, gt_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        rois = rois.view(-1, 5)
        rois_label = rois_label.view(-1).long()
        #gt_assign = gt_assign.view(-1).long()
        pos_id = rois_label.nonzero().squeeze()
        #gt_assign_pos = gt_assign[pos_id]
        #rois_label_pos = rois_label[pos_id]
        #rois_label_pos_ids = pos_id

        #rois_pos = Variable(rois[pos_id])
        rois = Variable(rois)
        rois_label = Variable(rois_label)

        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        roi_pool_feat = self._PyramidRoI_Feat(features, rois, im_info)
        #print(roi_pool_feat.shape)
        pooled_feat = self.rcnnfeatures(roi_pool_feat)
        #print(pooled_feat.shape)
        pooled_feat = pooled_feat.view(-1, self.rcnn_din)
        #print(pooled_feat.shape)
        cls_score = self.rcnn_cls(pooled_feat)
        bbox_pred = self.rcnn_bbox_pred(pooled_feat)
        cls_prob = F.softmax(cls_score)
        loss_cls = F.cross_entropy(cls_score, rois_label)
        # loss (l1-norm) for bounding box regression
        loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
        rois_label = rois_label.view(batch_size, -1)
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, rois_label
      else:
        rois = self.proposals((rpn_cls_prob_alls.data, rpn_bbox_pred_alls.data,
                                 im_info, 'TEST', rpn_shapes))
        rois = Variable(rois)

        roi_pool_feat = self._PyramidRoI_Feat(features, rois, im_info)
        pooled_feat = self.rcnnfeatures(roi_pool_feat)
        pooled_feat = pooled_feat.view(-1, self.rcnn_din)
        cls_score = self.rcnn_cls(pooled_feat)
        bbox_pred = self.rcnn_bbox_pred(pooled_feat)
        cls_prob = F.softmax(cls_score)
        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
        return rois, cls_prob, bbox_pred

def resnet18(num_clssses, training, pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = resnet_fpn(num_clssses, BasicBlock, [2, 2, 2, 2], training)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model

def resnet34(num_clssses, training, pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = resnet_fpn(num_clssses, BasicBlock, [3, 4, 6, 3], training)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='/home/user/.torch/models/'), strict=False)
  return model

def resnet50(num_clssses, training, pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = resnet_fpn(num_clssses, Bottleneck, [3, 4, 6, 3], training)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='/home/user/.torch/models/'), strict=False)
  return model

def resnet101(num_clssses, training, pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = resnet_fpn(num_clssses, Bottleneck, [3, 4, 23, 3], training)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='/home/user/.torch/models/'), strict=False)
  return model

def resnet152(num_clssses, training, pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = resnet_fpn(num_clssses, Bottleneck, [3, 8, 36, 3], training)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='/home/user/.torch/models/'), strict=False)
  return model
