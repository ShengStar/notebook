# pointpillar相关博客内容

https://github.com/SmallMunich/nutonomy_pointpillars

参考：https://blog.csdn.net/qq_39732684/article/details/105226255

参考：https://blog.csdn.net/LimitOut/article/details/108739865

https://blog.csdn.net/weixin_40805392/article/details/102135201?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param

https://blog.csdn.net/qq_39732684/article/details/105226255

# 3D-Dection系列论文1：Pointpillars ---example是如何生成的？：

# https://blog.csdn.net/LimitOut/article/details/108739865

1.anchor的生成：
　target_assigner.generate_anchors(feature_map_size)进入的我们的anchor生成，中间步骤挺繁琐…，最终生成1ｘ248ｘ216ｘ2=107136个anchor,每个anchor有７个值，长宽高，中心点，航向，返回的anchors＝(1, 248, 216, 2, 7)，最后reshape成<class ‘tuple’>: (107136, 7)的形式,这里面有bev_anchor的生成，就是在俯视图上划分的anchor，只需要最左上和最右下两个坐标就可以了．
　令大家需要知道的是dataset = KittiDataset(…)这个函数返回的是含有pkl信息的kitti_infos和pre_func函数，这里面还有一个内置函数：
————————————————
版权声明：本文为CSDN博主「LimitOut」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/LimitOut/article/details/108739865

https://blog.csdn.net/qq_39732684/article/details/105226255

```python
	# 进过上述讨论，x 是 [B, C, 200, 176] 的张量 
    def forward(self, x):
    	# conv_box 和 conv_cls 是 1*1 的卷积
        box_preds = self.conv_box(x) # 输出 [B, 14, 200, 176] 的张量 
        cls_preds = self.conv_cls(x) # 输出 [B, 2, 200, 176] 的张量
		# 为啥会出现 14？
		# 是因为 conv_box 的通道数定义为 num_anchor_per_loc * box_code_size = 2*7
 
        # 对张量做置换，contiguous 是让置换后的张量内存分布连续的操作
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous() # [B, 200, 176, 14]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous() # [B, 200, 176, 2]

        if self._use_direction_classifier:
        	# conv_dir_cls 也是 1*1 的卷积
            dir_cls_preds = self.conv_dir_cls(x) # 输出 [B, 4, 200, 176] 的张量
            # 为什么是 4 呢？
            # 是因为 conv_dir_cls 的通道数定义为 num_anchor_per_loc * 2 = 2*2
            # 输出 [B, 200, 176, 4] 的张量
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        return box_preds, cls_preds, dir_cls_preds

```

# rpn_head

简而言之，rpn_head预测3d目标框和它本身朝向以及它对应的类别，所以它的误差函数主要是两种：

3d目标框的误差函数；使用smooth L1误差损失函数；
3d目标框对应目标类别的误差函数；类别用one_hot编码，使用Focal loss损失函数；因为SA-SSD只识别车这一类，所以one_hot是2 × 1 2\times 12×1向量，表示车类点和背景点；
3d目标框朝向的误差函数；使用交叉熵损失函数；
rpn_head的前向计算并不难理解。比较复杂的是它误差计算的流程。复杂并不是说误差函数很复杂，而是其中的一些代码细节比较繁杂（有些细节我没有看懂，请多多见谅，日后看懂会更新）。我来看看它的loss函数：

```python
	# box_preds, cls_preds 是 RPN 网络的预测值，分别是
	# [B, 200, 176, 14] 的张量和 [B, 200, 176, 2] 的张量
	# gt_bboxes, gt_labels 是 3d 目标的框参数和类别
	# anchors, anchors_mask 的概念在上一篇博客已经介绍了
	# cfg 是配置参数
    def loss(self, box_preds, cls_preds, dir_cls_preds, gt_bboxes, gt_labels, anchors, anchors_mask, cfg):

        batch_size = box_preds.shape[0]

		# 下面几行代码的作用
		# 生成与 box_preds, cls_preds 相对应的真值 targets，cls_targets
		# 和与之对应的权值 reg_weights 和 cls_weights
        labels, targets, ious = multi_apply(create_target_torch,
                                            anchors, gt_bboxes,
                                            anchors_mask, gt_labels,
                                            similarity_fn=getattr(iou3d_utils, cfg.assigner.similarity_fn)(),
                                            box_encoding_fn = second_box_encode,
                                            matched_threshold=cfg.assigner.pos_iou_thr,
                                            unmatched_threshold=cfg.assigner.neg_iou_thr,
                                            box_code_size=self._box_code_size)


        labels = torch.stack(labels,)
        targets = torch.stack(targets)

		# 计算权值，计算方式跟辅助网络中的很相似
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels)

		# cared 表示 labels >= 0 的 bool 张量
		# cls_targets 就是过滤掉 labels == -1 的张量
        cls_targets = labels * cared.type_as(labels)

		# 根据预测值，真值，权重，构建误差函数
		# 为了让 3d框 的回归变得更加准确，加入 _encode_rad_error_by_sin 更细致刻画 3d 框
		# loc_loss 是 3d框 的误差
		# cls_loss 是 3d框类别 的误差
		# 权值的意义：
		# 对于 loc_loss，我只关心车这一类的3d目标框，设置其他类和背景点的权值为零，滤除它们
		# 对于 cls_loss，正样本和负样本数量差异太大，比如正样本（是车的目标）太少，
		# 需要加大它误差对应的权值，提高网络对车识别的准确率
        loc_loss, cls_loss = self.create_loss(
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights,
            reg_targets=targets,
            reg_weights=reg_weights,
            num_class=self._num_class,
            encode_rad_error_by_sin=self._encode_rad_error_by_sin,
            use_sigmoid_cls=self._use_sigmoid_cls,
            box_code_size=self._box_code_size,
        )

        loc_loss_reduced = loc_loss / batch_size
        loc_loss_reduced *= 2

        cls_loss_reduced = cls_loss / batch_size
        cls_loss_reduced *= 1

        loss = loc_loss_reduced + cls_loss_reduced

        if self._use_direction_classifier:
        	# 生成与 dir_cls_preds 对应的真值 dir_labels 
            dir_labels = self.get_direction_target(anchors, targets, use_one_hot=False).view(-1)
            dir_logits = dir_cls_preds.view(-1, 2)
            # 设置权值是为了仅仅考虑 labels > 0 的目标（即车这一类）
            weights = (labels > 0).type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 使用交叉熵做朝向预测的误差损失函数
            dir_loss = weighted_cross_entropy(dir_logits, dir_labels,
                                              weight=weights.view(-1),
                                              avg_factor=1.)

            dir_loss_reduced = dir_loss / batch_size
            dir_loss_reduced *= .2
            loss += dir_loss_reduced

        return dict(rpn_loc_loss=loc_loss_reduced, rpn_cls_loss=cls_loss_reduced, rpn_dir_loss=dir_loss_reduced)

```

https://blog.csdn.net/qq_39732684/article/details/105188258?spm=1001.2014.3001.5501

## Anchor的作用

3d目标检测预测一个目标的7个参数外加目标的类别（共8个参数）。假设我只预测车这一类，那么我需要回归出一个目标的7个参数，即xyzwlh和yaw角。然而网络大多是不靠谱的，它回归出来一堆不太精确的目标。考虑到车这一类有着共性，比如各色型号的车的长宽高都差不多（专指小车），以及车都在地上跑（车中心距离地面的高度差不多一致）。Anchor是作为3d目标的一种先验（Prior），指3d目标可能以某种姿态角度出现的地方。如果我只识别车，我可以生成一堆Anchors，固定它的wlh和z，让它们匀称地分布在BEV视图下。3d目标一定在某个Anchor的附近。给不靠谱网络识别的3D框和这一堆Anchors做类似交集的运算，可以得到一些靠谱的Anchors（即SA-SSD中的引导Anchor），用于做后续处理。

有时候，Anchors的数量太多了。考虑到有点云的地方才会有目标，我们可以扔掉那些自身不覆盖任何点云的Anchor（这是Anchor Mask的工作）。然后对剩下的Acnhor和不靠谱网络生成的3D框做类似交集的运算，可以得到一些靠谱的Anchors（即SA-SSD中的引导Anchor），用于做后续处理。

