import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import DataLoader
from helper import getCoef,sample_gaussian_2d

def denormalize(normalized_value, feature_type, norm_params):
    # 修改后的逻辑
    if feature_type == 'position_x':
        min_val, max_val = norm_params['position']['x']
    elif feature_type == 'position_y':
        min_val, max_val = norm_params['position']['y']
    elif feature_type == 'heading':
        min_val, max_val = norm_params['heading']
    else:
        raise KeyError(f"特征类型 {feature_type} 不在参数中")

    # 反归一化公式（与之前的归一化公式对应）
    return (normalized_value + 1) * (max_val - min_val) / 2 + min_val


def Gaussian2DLikelihood(outputs, targets, nodesPresent, obs_length, seq_length, dataset_index, args,nodes,norm_params):
    """
    完整轨迹预测损失函数，包含：
    - 原始高斯似然损失
    - 航向角损失
    - 动力学约束损失
    - 运动状态分布一致性损失（基于核密度估计）
    """
    # 获取分布参数
    mux, muy, sx, sy, corr = getCoef(outputs)
    next_x, next_y = sample_gaussian_2d(
        mux.data,
        muy.data,
        sx.data,
        sy.data,
        corr.data,
        nodesPresent[args.obs_length - 1],
    )
    # 反归一化真实值 (nodes)
    nodes_real_denorm = nodes.clone()
    nodes_real_denorm[:, :, 0] = denormalize(nodes[:, :, 0], 'position_x', norm_params)
    nodes_real_denorm[:, :, 1] = denormalize(nodes[:, :, 1], 'position_y', norm_params)
    nodes_real_denorm[:, :, 2] = denormalize(nodes[:, :, 2], 'heading', norm_params)

    # 反归一化预测值 (outputs)
    outputs_denorm = outputs.clone()
    outputs_denorm[:, :, 0] = denormalize(next_x, 'position_x', norm_params)  # mux
    outputs_denorm[:, :, 1] = denormalize(next_y, 'position_y', norm_params)  # muy
    outputs_denorm[:, :, 5] = denormalize(outputs[:, :, 5], 'heading', norm_params)  # pred_heading
    print("反归一化验证:")
    print(
        f"预测x范围: [{outputs_denorm[:, :, 0].min().item():.2f}, {outputs_denorm[:, :, 0].max().item():.2f}]")
    print(
        f"真实x范围: [{nodes_real_denorm[:, :, 0].min().item():.2f}, {nodes_real_denorm[:, :, 0].max().item():.2f}]")

    # 原始节点存在信息处理（保留原始逻辑）
    nodesPresent = [[m[0] for m in t] for t in nodesPresent]
    seq_length = seq_length[dataset_index]
    device = outputs.device
    device = outputs_denorm.device

    # ==================== 原始高斯似然计算 ====================
    # 计算高斯概率密度（保持原始公式）
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (
            torch.pow((normx / sx), 2)
            + torch.pow((normy / sy), 2)
            - 2 * ((corr * normx * normy) / sxsy)
    )
    negRho = 1 - torch.pow(corr, 2)

    # 数值稳定性处理（保留原始实现）
    result = torch.exp(-z / (2 * negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    epsilon = 1e-20
    result_pos = -torch.log(torch.clamp(result / denom, min=epsilon))

    # ==================== 航向损失计算 ====================
    # 新增航向角损失（保留用户原始实现）
    pred_heading = outputs[:, :, 5]
    target_heading = targets[:, :, 2]
    heading_loss = torch.abs(pred_heading - target_heading)

    # ==================== 动力学约束计算 ====================
    # 运动学参数计算（保留用户原始实现）
    theta = outputs_denorm[:, :, 5]

    # 速度计算（注意时间步范围）
    v_x = outputs_denorm[1:, :, 0] - outputs_denorm[:-1, :, 0]
    v_y = outputs_denorm[1:, :, 1] - outputs_denorm[:-1, :, 1]

    # 加速度计算（保持原始实现）
    a_x = v_x[1:] - v_x[:-1]
    a_y = v_y[1:] - v_y[:-1]

    # 分解加速度到车体坐标系
    theta_for_a = theta[1:-1, :]
    a_lon = a_x * torch.cos(theta_for_a) + a_y * torch.sin(theta_for_a)
    a_lat = -a_x * torch.sin(theta_for_a) + a_y * torch.cos(theta_for_a)

    # 加加速度计算（保留用户实现）
    jerk_x = a_x[1:] - a_x[:-1]
    jerk_y = a_y[1:] - a_y[:-1]
    theta_for_jerk = theta[2:-1, :]
    jerk_lon = jerk_x * torch.cos(theta_for_jerk) + jerk_y * torch.sin(theta_for_jerk)
    jerk_lat = -jerk_x * torch.sin(theta_for_jerk) + jerk_y * torch.cos(theta_for_jerk)

    # 横摆角速度
    yaw_rate = theta[1:] - theta[:-1]

    # ==================== 可微分分布一致性计算 ====================
    def kde_loss(pred, real, num_bins=20, sigma=0.1):
        """
        基于核密度估计的分布差异计算
        参数：
            pred: 预测值 (N,)
            real: 真实值 (M,)
            num_bins: 分布分箱数
            sigma: 高斯核带宽
        返回：
            kl_loss: 可微分的KL散度损失
        """
        # 合并样本确定范围
        combined = torch.cat([pred.detach(), real.detach()])
        min_val = combined.min().item()
        max_val = combined.max().item()

        # 处理全零范围的情况
        if max_val - min_val < 1e-6:
            return torch.tensor(0.0, device=device)

        # 生成可微分的bin中心
        bin_centers = torch.linspace(min_val, max_val, num_bins, device=device)

        # 计算核密度估计（可微分操作）
        def kde(x):
            # 扩展维度计算差值 [x元素数, bin数]
            diff = x.unsqueeze(1) - bin_centers.unsqueeze(0)
            # 高斯核计算 [x元素数, bin数]
            weights = torch.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            return weights.mean(dim=0)  # 沿样本维度平均

        # 计算概率分布
        pred_dist = kde(pred) + 1e-8
        real_dist = kde(real) + 1e-8

        # 正则化概率分布
        pred_dist = pred_dist / pred_dist.sum()
        real_dist = real_dist / real_dist.sum()

        # 计算反向KL散度（更稳定）
        return F.kl_div(torch.log(pred_dist), real_dist, reduction='sum')

    # ==================== 运动特征提取 ====================
    seq_len, num_nodes = nodes_real_denorm.shape[:2]
    valid_mask = torch.zeros(seq_len, num_nodes, dtype=torch.bool, device=device)
    for t in range(seq_len):
        valid_nodes = [n for n in nodesPresent[t] if n < num_nodes]
        valid_mask[t, valid_nodes] = True

    # 真实运动特征
    real_v = nodes_real_denorm[1:, :, :2] - nodes_real_denorm[:-1, :, :2]
    real_speed = torch.norm(real_v, dim=2)
    speed_valid = valid_mask[1:] & valid_mask[:-1]

    real_a = real_v[1:] - real_v[:-1]
    real_heading = nodes_real_denorm[1:-1, :, 2]
    real_a_lon = (real_a[..., 0] * torch.cos(real_heading) +
                  real_a[..., 1] * torch.sin(real_heading))
    a_lon_valid = valid_mask[1:-1] & valid_mask[:-2] & valid_mask[2:]

    real_yaw = nodes_real_denorm[1:, :, 2] - nodes_real_denorm[:-1, :, 2]
    yaw_valid = valid_mask[1:] & valid_mask[:-1]

    # 预测运动特征
    pred_speed = torch.norm(outputs_denorm[1:, :, :2] - outputs_denorm[:-1, :, :2], dim=2)
    pred_a_lon = a_lon
    pred_yaw = yaw_rate

    # ==================== 分布一致性损失计算 ====================
    loss_B = 0.0
    weight = 20.0 / 3  # 每个B子指标占20/3分

    # 速度分布（B1）
    if torch.sum(speed_valid) > 10:
        loss_B += kde_loss(pred_speed[speed_valid],
                           real_speed[speed_valid],
                           sigma=0.2) * weight

    # 纵向加速度分布（B2）
    if torch.sum(a_lon_valid) > 10:
        loss_B += kde_loss(pred_a_lon[a_lon_valid],
                           real_a_lon[a_lon_valid],
                           sigma=0.1) * weight

    # 角速度分布（B3）
    if torch.sum(yaw_valid) > 10:
        loss_B += kde_loss(pred_yaw[yaw_valid],
                           real_yaw[yaw_valid],
                           sigma=0.05) * weight

    # ==================== 损失累计循环 ====================
    loss_pos = 0.0
    loss_heading = 0.0
    loss_dynamics = 0.0
    counter = 0

    for framenum in range(obs_length + 1, seq_length):
        nodeIDs = nodesPresent[framenum]
        for nodeID in nodeIDs:
            # 原始位置损失
            loss_pos += result_pos[framenum, nodeID]

            # 航向损失
            loss_heading += heading_loss[framenum, nodeID]

            # 动力学约束（保留原始调试逻辑）
            current_loss = 0.0
            # 检查纵向和横向加速度
            if 1 <= framenum <= seq_length - 2:
                a_lon_val = a_lon[framenum - 1, nodeID]
                a_lat_val = a_lat[framenum - 1, nodeID]
                current_loss += torch.clamp(torch.abs(a_lon_val) - 3.0, min=0)
                current_loss += torch.clamp(torch.abs(a_lat_val) - 0.5, min=0)

            # 检查加加速度
            if 2 <= framenum <= seq_length - 3:
                jerk_lon_val = jerk_lon[framenum - 2, nodeID]
                jerk_lat_val = jerk_lat[framenum - 2, nodeID]
                current_loss += torch.clamp(torch.abs(jerk_lon_val) - 6.0, min=0)
                current_loss += torch.clamp(torch.abs(jerk_lat_val) - 1.0, min=0)

            # 检查横摆角速度
            if 1 <= framenum <= seq_length - 2:
                yaw_val = yaw_rate[framenum - 1, nodeID]
                current_loss += torch.clamp(torch.abs(yaw_val) - 0.5, min=0)

            loss_dynamics += current_loss
            counter += 1

    # ==================== 损失整合 ====================
    if counter > 0:
        # 平均各损失项
        loss_pos /= counter
        loss_heading /= counter
        loss_dynamics /= counter
        loss_B /= 3  # 三个子指标平均

        # 总损失计算（动态权重平衡）
        total_loss = loss_pos + loss_heading + 0.1 * loss_dynamics + 0.05 * loss_B

        # 调试输出（保留原始格式）
        print('[Loss Breakdown]')
        print(f'  Position: {loss_pos:.4f} (原始高斯损失)')
        print(f'  Heading: {loss_heading:.4f} (航向角损失)')
        print(f'  Dynamics: {loss_dynamics:.4f} (动力学约束)')
        print(f'  B-Consistency: {loss_B:.4f} (运动分布一致性)')
        print(f'  Total: {total_loss:.4f}\n')

        return total_loss
    else:
        return 0.0



# # 增加动力学约束损失函数
# def Gaussian2DLikelihood(outputs, targets, nodesPresent, obs_length, seq_length, dataset_index):
#     nodesPresent = [[m[0] for m in t] for t in nodesPresent]
#     # Get the observed length
#     seq_length = seq_length[dataset_index]
#
#     # Extract mean, std devs and correlation
#     mux, muy, sx, sy, corr = getCoef(outputs)
#
#     # Compute factors
#     normx = targets[:, :, 0] - mux
#     normy = targets[:, :, 1] - muy
#     sxsy = sx * sy
#     z = (
#         torch.pow((normx / sx), 2)
#         + torch.pow((normy / sy), 2)
#         - 2 * ((corr * normx * normy) / sxsy)
#     )
#     negRho = 1 - torch.pow(corr, 2)
#
#     # Numerator
#     result = torch.exp(-z / (2 * negRho))
#
#     # Normalization factor
#     denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
#     result = result / denom
#     epsilon = 1e-20
#     result_pos = -torch.log(torch.clamp(result, min=epsilon))
#
#     # Compute heading loss 增加航向损失
#     pred_heading = outputs[:, :, 5]
#     target_heading = targets[:, :, 2]
#     heading_loss = torch.abs(pred_heading - target_heading)# 简单的绝对误差损失，可根据需求调整
#
#    # Compute the loss across all frames and all nodes
#     loss_pos = 0
#     loss_heading = 0
#     loss_dynamics = 0  # 动力学损失
#     counter = 0
#
#     # 预先计算动力学参数
#     theta = outputs[:, :, 5]
#     # 速度
#     v_x = outputs[1:, :, 0] - outputs[:-1, :, 0]
#     v_y = outputs[1:, :, 1] - outputs[:-1, :, 1]
#     # 加速度
#     a_x = v_x[1:, :] - v_x[:-1, :]
#     a_y = v_y[1:, :] - v_y[:-1, :]
#     # 加加速度
#     jerk_x = a_x[1:, :] - a_x[:-1, :]
#     jerk_y = a_y[1:, :] - a_y[:-1, :]
#     # 横摆角速度
#     yaw_rate = theta[1:, :] - theta[:-1, :]
#     # 分解加速度到纵向和横向
#     theta_for_a = theta[1:-1, :]
#     a_lon = a_x * torch.cos(theta_for_a) + a_y * torch.sin(theta_for_a)
#     a_lat = -a_x * torch.sin(theta_for_a) + a_y * torch.cos(theta_for_a)
#     # 分解加加速度
#     theta_for_jerk = theta[2:-1, :]
#     jerk_lon = jerk_x * torch.cos(theta_for_jerk) + jerk_y * torch.sin(theta_for_jerk)
#     jerk_lat = -jerk_x * torch.sin(theta_for_jerk) + jerk_y * torch.cos(theta_for_jerk)
#
#     for framenum in range(obs_length+1, seq_length):
#         nodeIDs = nodesPresent[framenum]
#         for nodeID in nodeIDs:
#             # 位置损失
#             loss_pos += result_pos[framenum, nodeID]
#             # 航向损失
#             loss_heading += heading_loss[framenum, nodeID]
#             # 动力学损失
#             current_dynamics_loss = 0
#             # 检查纵向和横向加速度
#             if framenum >= 1 and framenum <= (seq_length - 2):
#                 a_lon_val = a_lon[framenum - 1, nodeID]
#                 a_lat_val = a_lat[framenum - 1, nodeID]
#                 current_dynamics_loss += torch.clamp(torch.abs(a_lon_val) - 3, min=0)
#                 current_dynamics_loss += torch.clamp(torch.abs(a_lat_val) - 0.5, min=0)
#             # 检查加加速度
#             if framenum >= 2 and framenum <= (seq_length - 3):
#                 jerk_lon_val = jerk_lon[framenum - 2, nodeID]
#                 jerk_lat_val = jerk_lat[framenum - 2, nodeID]
#                 current_dynamics_loss += torch.clamp(torch.abs(jerk_lon_val) - 6, min=0)
#                 current_dynamics_loss += torch.clamp(torch.abs(jerk_lat_val) - 1, min=0)
#             # 检查横摆角速度
#             if framenum >= 1 and framenum <= (seq_length - 2):
#                 yaw_val = yaw_rate[framenum - 1, nodeID]
#                 current_dynamics_loss += torch.clamp(torch.abs(yaw_val) - 0.5, min=0)
#
#
#             # ========== 调试代码插入位置 ========== #
#             current_dynamics_loss = 0
#             debug_info = []  # 用于记录调试信息
#
#             # 检查纵向和横向加速度
#             if framenum >= 1 and framenum <= (seq_length - 2):
#                 a_lon_val = a_lon[framenum - 1, nodeID]
#                 a_lat_val = a_lat[framenum - 1, nodeID]
#
#                 # 记录原始值和约束触发量
#                 debug_info.append(f"Frame={framenum}, Node={nodeID}")
#                 debug_info.append(f"  a_lon={a_lon_val.item():.4f} (阈值±3)")
#                 debug_info.append(f"  a_lat={a_lat_val.item():.4f} (阈值±0.5)")
#
#                 # 计算约束损失
#                 a_lon_loss = torch.clamp(torch.abs(a_lon_val) - 3, min=0)
#                 a_lat_loss = torch.clamp(torch.abs(a_lat_val) - 0.5, min=0)
#                 current_dynamics_loss += a_lon_loss + a_lat_loss
#                 debug_info.append(f"  a_loss: lon={a_lon_loss.item():.4f}, lat={a_lat_loss.item():.4f}")
#
#             # 检查加加速度
#             if framenum >= 2 and framenum <= (seq_length - 3):
#                 jerk_lon_val = jerk_lon[framenum - 2, nodeID]
#                 jerk_lat_val = jerk_lat[framenum - 2, nodeID]
#
#                 debug_info.append(f"  jerk_lon={jerk_lon_val.item():.4f} (阈值±6)")
#                 debug_info.append(f"  jerk_lat={jerk_lat_val.item():.4f} (阈值±1)")
#
#                 jerk_lon_loss = torch.clamp(torch.abs(jerk_lon_val) - 6, min=0)
#                 jerk_lat_loss = torch.clamp(torch.abs(jerk_lat_val) - 1, min=0)
#                 current_dynamics_loss += jerk_lon_loss + jerk_lat_loss
#                 debug_info.append(f"  jerk_loss: lon={jerk_lon_loss.item():.4f}, lat={jerk_lat_loss.item():.4f}")
#
#             # 检查横摆角速度
#             if framenum >= 1 and framenum <= (seq_length - 2):
#                 yaw_val = yaw_rate[framenum - 1, nodeID]
#
#                 debug_info.append(f"  yaw_rate={yaw_val.item():.4f} (阈值±0.5)")
#                 yaw_loss = torch.clamp(torch.abs(yaw_val) - 0.5, min=0)
#                 current_dynamics_loss += yaw_loss
#                 debug_info.append(f"  yaw_loss={yaw_loss.item():.4f}")
#
#             # 打印调试信息（仅在触发约束时打印）
#             if current_dynamics_loss > 0:
#                 print("\n".join(debug_info))
#
#             # ========== 调试代码插入位置 ========== #
#
#             loss_dynamics += current_dynamics_loss
#             counter += 1
#
#     if counter != 0:
#         loss_pos /= counter
#         loss_heading /= counter
#         loss_dynamics /= counter
#         # 总损失，动力学权重可根据需要调整
#         total_loss = loss_pos + loss_heading + 0.1 * loss_dynamics
#         print(f'Losses - Pos: {loss_pos:.4f}, Heading: {loss_heading:.4f}, Dynamics: {loss_dynamics:.4f}')
#         return total_loss
#     else:
#         return 0



# # 迭代预测损失函数
# def Gaussian2DLikelihood(outputs, targets, nodesPresent, obs_length,seq_length,dataset_index):
#     """
#     Computes the likelihood of predicted locations under a bivariate Gaussian distribution
#     params:
#     outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size #[20, 10, 5]
#     targets: Torch variable containing tensor of shape seq_length x numNodes x input_size #[20, 10, 2]
#     nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame #len = 20
#     """
#     nodesPresent = [[m[0] for m in t] for t in nodesPresent]
#     # Get the sequence length
#
#
#     # Get the observed length
#     # dataloader = DataLoader(batch_size=1, obs_length=32, forcePreProcess=True)
#     # seq_length = dataloader.seq_length + 1
#     seq_length = seq_length[dataset_index]
#
#     # Extract mean, std devs and correlation
#     mux, muy, sx, sy, corr = getCoef(outputs)
#     # print(f"Shape of targets[:, :, 0]: {targets[:, :, 0].shape}")
#     # print(f"Shape of mux: {mux.shape}")
#
#     # Compute factors
#     normx = targets[:, :, 0] - mux
#     normy = targets[:, :, 1] - muy
#     sxsy = sx * sy
#     z = (
#         torch.pow((normx / sx), 2)
#         + torch.pow((normy / sy), 2)
#         - 2 * ((corr * normx * normy) / sxsy)
#     )
#     negRho = 1 - torch.pow(corr, 2)
#
#     # Numerator
#     result = torch.exp(-z / (2 * negRho))
#     # Normalization factor
#     denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
#
#     # Final PDF calculation
#     result = result / denom
#
#     # Numerical stability
#     epsilon = 1e-20
#     result_pos = -torch.log(torch.clamp(result, min=epsilon))
#
#     # Compute heading loss 增加航向损失
#     pred_heading = outputs[:, :, 5]
#     target_heading = targets[:, :, 2]
#     heading_loss = torch.abs(pred_heading - target_heading)  # 简单的绝对误差损失，可根据需求调整
#
#     # Compute the loss across all frames and all nodes
#     loss_pos = 0
#     loss_heading = 0
#     counter = 0
#
#     for framenum in range(obs_length+1, seq_length):
#         nodeIDs = nodesPresent[framenum]
#
#         for nodeID in nodeIDs:
#             loss_pos = loss_pos + result_pos[framenum, nodeID]
#             loss_heading = loss_heading + heading_loss[framenum, nodeID]
#             counter = counter + 1
#             # print('第{}帧----第{}个代理----位置损失：{}----航向损失：{}'.format(framenum,nodeID,loss_pos,loss_heading))
#
#     if counter != 0:
#         loss_pos = loss_pos / counter
#         loss_heading = loss_heading / counter
#         print('loss_pos:{}-----loss_heading:{}'.format(loss_pos,loss_heading))
#         total_loss = loss_pos + loss_heading  # 可根据需求调整权重
#         return total_loss
#     else:
#         return 0


"""
def Gaussian2DLikelihoodInference(outputs, targets, assumedNodesPresent, nodesPresent, use_cuda):
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time
    params:
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    
    # Extract mean, std devs and correlation
    #assumedNodesPresent =
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    # Compute the loss
    loss = Variable(torch.zeros(1))
    if use_cuda:
        loss = loss.cuda()
    counter = 0

    for framenum in range(outputs.size()[0]):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:
            if nodeID not in assumedNodesPresent:
                # If the node wasn't assumed to be present, don't compute loss for it
                continue
            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss       
"""
