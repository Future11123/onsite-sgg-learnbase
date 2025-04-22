import numpy as np
import torch
from torch.autograd import Variable
from utils import DataLoader
from helper import getCoef


# 增加动力学约束损失函数
def Gaussian2DLikelihood(outputs, targets, nodesPresent, obs_length, seq_length, dataset_index):
    nodesPresent = [[m[0] for m in t] for t in nodesPresent]
    # Get the observed length
    seq_length = seq_length[dataset_index]

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (
        torch.pow((normx / sx), 2)
        + torch.pow((normy / sy), 2)
        - 2 * ((corr * normx * normy) / sxsy)
    )
    negRho = 1 - torch.pow(corr, 2)

    # Numerator
    result = torch.exp(-z / (2 * negRho))

    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    result = result / denom
    epsilon = 1e-20
    result_pos = -torch.log(torch.clamp(result, min=epsilon))

    # Compute heading loss 增加航向损失
    pred_heading = outputs[:, :, 5]
    target_heading = targets[:, :, 2]
    heading_loss = torch.abs(pred_heading - target_heading)# 简单的绝对误差损失，可根据需求调整

   # Compute the loss across all frames and all nodes
    loss_pos = 0
    loss_heading = 0
    loss_dynamics = 0  # 动力学损失
    counter = 0

    # 预先计算动力学参数
    theta = outputs[:, :, 5]
    # 速度
    v_x = outputs[1:, :, 0] - outputs[:-1, :, 0]
    v_y = outputs[1:, :, 1] - outputs[:-1, :, 1]
    # 加速度
    a_x = v_x[1:, :] - v_x[:-1, :]
    a_y = v_y[1:, :] - v_y[:-1, :]
    # 加加速度
    jerk_x = a_x[1:, :] - a_x[:-1, :]
    jerk_y = a_y[1:, :] - a_y[:-1, :]
    # 横摆角速度
    yaw_rate = theta[1:, :] - theta[:-1, :]
    # 分解加速度到纵向和横向
    theta_for_a = theta[1:-1, :]
    a_lon = a_x * torch.cos(theta_for_a) + a_y * torch.sin(theta_for_a)
    a_lat = -a_x * torch.sin(theta_for_a) + a_y * torch.cos(theta_for_a)
    # 分解加加速度
    theta_for_jerk = theta[2:-1, :]
    jerk_lon = jerk_x * torch.cos(theta_for_jerk) + jerk_y * torch.sin(theta_for_jerk)
    jerk_lat = -jerk_x * torch.sin(theta_for_jerk) + jerk_y * torch.cos(theta_for_jerk)

    for framenum in range(obs_length+1, seq_length):
        nodeIDs = nodesPresent[framenum]
        for nodeID in nodeIDs:
            # 位置损失
            loss_pos += result_pos[framenum, nodeID]
            # 航向损失
            loss_heading += heading_loss[framenum, nodeID]
            # 动力学损失
            current_dynamics_loss = 0
            # 检查纵向和横向加速度
            if framenum >= 1 and framenum <= (seq_length - 2):
                a_lon_val = a_lon[framenum - 1, nodeID]
                a_lat_val = a_lat[framenum - 1, nodeID]
                current_dynamics_loss += torch.clamp(torch.abs(a_lon_val) - 3, min=0)
                current_dynamics_loss += torch.clamp(torch.abs(a_lat_val) - 0.5, min=0)
            # 检查加加速度
            if framenum >= 2 and framenum <= (seq_length - 3):
                jerk_lon_val = jerk_lon[framenum - 2, nodeID]
                jerk_lat_val = jerk_lat[framenum - 2, nodeID]
                current_dynamics_loss += torch.clamp(torch.abs(jerk_lon_val) - 6, min=0)
                current_dynamics_loss += torch.clamp(torch.abs(jerk_lat_val) - 1, min=0)
            # 检查横摆角速度
            if framenum >= 1 and framenum <= (seq_length - 2):
                yaw_val = yaw_rate[framenum - 1, nodeID]
                current_dynamics_loss += torch.clamp(torch.abs(yaw_val) - 0.5, min=0)


            # ========== 调试代码插入位置 ========== #
            current_dynamics_loss = 0
            debug_info = []  # 用于记录调试信息

            # 检查纵向和横向加速度
            if framenum >= 1 and framenum <= (seq_length - 2):
                a_lon_val = a_lon[framenum - 1, nodeID]
                a_lat_val = a_lat[framenum - 1, nodeID]

                # 记录原始值和约束触发量
                debug_info.append(f"Frame={framenum}, Node={nodeID}")
                debug_info.append(f"  a_lon={a_lon_val.item():.4f} (阈值±3)")
                debug_info.append(f"  a_lat={a_lat_val.item():.4f} (阈值±0.5)")

                # 计算约束损失
                a_lon_loss = torch.clamp(torch.abs(a_lon_val) - 3, min=0)
                a_lat_loss = torch.clamp(torch.abs(a_lat_val) - 0.5, min=0)
                current_dynamics_loss += a_lon_loss + a_lat_loss
                debug_info.append(f"  a_loss: lon={a_lon_loss.item():.4f}, lat={a_lat_loss.item():.4f}")

            # 检查加加速度
            if framenum >= 2 and framenum <= (seq_length - 3):
                jerk_lon_val = jerk_lon[framenum - 2, nodeID]
                jerk_lat_val = jerk_lat[framenum - 2, nodeID]

                debug_info.append(f"  jerk_lon={jerk_lon_val.item():.4f} (阈值±6)")
                debug_info.append(f"  jerk_lat={jerk_lat_val.item():.4f} (阈值±1)")

                jerk_lon_loss = torch.clamp(torch.abs(jerk_lon_val) - 6, min=0)
                jerk_lat_loss = torch.clamp(torch.abs(jerk_lat_val) - 1, min=0)
                current_dynamics_loss += jerk_lon_loss + jerk_lat_loss
                debug_info.append(f"  jerk_loss: lon={jerk_lon_loss.item():.4f}, lat={jerk_lat_loss.item():.4f}")

            # 检查横摆角速度
            if framenum >= 1 and framenum <= (seq_length - 2):
                yaw_val = yaw_rate[framenum - 1, nodeID]

                debug_info.append(f"  yaw_rate={yaw_val.item():.4f} (阈值±0.5)")
                yaw_loss = torch.clamp(torch.abs(yaw_val) - 0.5, min=0)
                current_dynamics_loss += yaw_loss
                debug_info.append(f"  yaw_loss={yaw_loss.item():.4f}")

            # 打印调试信息（仅在触发约束时打印）
            if current_dynamics_loss > 0:
                print("\n".join(debug_info))

            # ========== 调试代码插入位置 ========== #

            loss_dynamics += current_dynamics_loss
            counter += 1

    if counter != 0:
        loss_pos /= counter
        loss_heading /= counter
        loss_dynamics /= counter
        # 总损失，动力学权重可根据需要调整
        total_loss = loss_pos + loss_heading + 0.1 * loss_dynamics
        print(f'Losses - Pos: {loss_pos:.4f}, Heading: {loss_heading:.4f}, Dynamics: {loss_dynamics:.4f}')
        return total_loss
    else:
        return 0



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
