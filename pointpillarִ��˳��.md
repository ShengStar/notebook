# pointpillar执行顺序

执行

1

```python
    if box_coder_type == 'ground_box3d_coder':
        cfg = box_coder_config.ground_box3d_coder
        print('执行')
        print(box_coder_type)
        return GroundBox3dCoderTorch(cfg.linear_dim, cfg.encode_angle_vector)
```

2

```python
class GroundBox3dCoderTorch(GroundBox3dCoder):
    def encode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_encode(boxes, anchors, self.vec_encode, self.linear_dim)

    def decode_torch(self, boxes, anchors):
        return box_torch_ops.second_box_decode(boxes, anchors, self.vec_encode, self.linear_dim)
```

3

```python
def second_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim=False):
    """box encode for VoxelNet
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, l, w, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
    xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    za = za + ha / 2
    zg = zg + hg / 2
    diagonal = torch.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return torch.cat([xt, yt, zt, wt, lt, ht, rtx, rty], dim=-1)
    else:
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

    # rt = rg - ra
    # return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)
```

参数

```python
target_assigner: {
      anchor_generators: {
         anchor_generator_stride: {
           sizes: [1.6, 3.9, 1.56] # wlh
           strides: [0.32, 0.32, 0.0] # if generate only 1 z_center, z_stride will be ignored
           offsets: [0.16, -39.52, -1.78] # origin_offset + strides / 2
           rotations: [0, 1.57] # 0, pi/2
           matched_threshold : 0.6
           unmatched_threshold : 0.45
         }
       }

      sample_positive_fraction : -1
      sample_size : 512
      region_similarity_calculator: {
        nearest_iou_similarity: {
        }
      }
    }
```

