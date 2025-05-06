import xml.dom.minidom
import D1xodrRoads
import matplotlib.pyplot as plt
import math
import numpy as np


def get_available_path(link_map, origin, destination):
    paths = []                      # 存储所有可用路径
    path = [origin]                 # 初始化路径，包含起点
    outlink_id_lst_pre = [origin]   # 初始化前驱路径列表，包含起点
    paths.append(path)              # 将初始路径添加到路径列表中
    path_length = 1                 # 最大路径长度，初始值为 1（即起点）

    # 当前驱路径列表不为空且路径长度小于 20 时继续遍历
    while outlink_id_lst_pre and path_length < 20:
        nextlink_id_lst = outlink_id_lst_pre    # 备份当前前驱路径列表
        paths_new = []                          # 用于存储新的路径列表
        outlink_id_lst_pre = []                 # 清空前驱路径列表，准备存储新的路径
        # 遍历前驱路径列表中的每个链接 ID
        for link_id in nextlink_id_lst:
            outlink_id_lst = []  # 临时存储后续链接 ID 列表
            # 获取当前链接的详细信息
            nextlink = link_map.get(link_id)
            if not nextlink or not nextlink.out_link_lst:  # 如果链接不存在或者没有输出链接，跳过
                continue
            # 遍历当前链接的所有输出链接
            for ll in nextlink.out_link_lst:
                # 如果该链接 ID 不在后续链接列表中且不等于起点，则添加到后续链接列表
                if ll.id not in outlink_id_lst and ll.id != origin:
                    outlink_id_lst.append(ll.id)
                    outlink_id_lst_pre.append(ll.id)  # 将后续链接添加到前驱路径列表
            # 遍历已有路径
            for path in paths:
                # 如果路径的最后一个节点等于当前链接 ID
                if path[-1] == link_id:
                    # 遍历后续链接，避免形成环路
                    for lid in outlink_id_lst:
                        # 如果后续链接 ID 已经在路径中，则跳过
                        if lid in path:
                            # 如果路径未在新路径列表中，则添加
                            if path not in paths_new:
                                paths_new.append(path)
                            continue
                        # 创建新的路径
                        path_new = path + [lid]
                        # 如果新路径不在路径列表中，则添加
                        if path_new not in paths_new:
                            paths_new.append(path_new)
        # 如果找到新的路径，则更新路径列表
        if paths_new:
            paths = paths_new
        path_length += 1  # 增加路径长度
        # 遍历所有路径，如果某条路径的最后一个节点是目标点，返回该路径
        for pathx in paths:
            if pathx[-1] == destination:
                return pathx
    # 如果没有找到可用路径，抛出异常
    raise Exception("No available path")

def find_legal_connections_between_o_d(link_map, o_points, d_points):
    legal_connections = []  # 存储合法的连接路径
    # 遍历所有的起点 o_points
    for origin_link_id in o_points:
        # 遍历所有的终点 d_points
        for destination_link_id in d_points:
            # 如果起点和终点相同，则跳过
            if origin_link_id == destination_link_id:
                continue
            try:
                # 获取起点到终点的可用路径
                path = get_available_path(link_map, origin_link_id, destination_link_id)
                # 如果路径不在合法连接中，则添加路径
                if path not in legal_connections:
                    legal_connections.append(path)
            except Exception:
                pass  # 如果获取路径时发生异常，忽略该连接
    return legal_connections  # 返回所有合法连接路径

def organize_legal_connections(legal_connections):
    organized_connections = []  # 存储组织后的连接路径
    # 遍历所有的合法连接路径
    for path in legal_connections:
        # 如果路径长度大于 2，则仅保留起点和终点
        if len(path) > 2:
            organized_connections.append([path[0], path[-1], 0])
        # 如果路径长度等于 2，直接保存该路径
        elif len(path) == 2:
            organized_connections.append([path[0], path[1], 0])
    return organized_connections  # 返回组织后的连接路径

def generate_basic_od_pairs(link_map):
    od_pairs = []                       # 存储 OD 对
    link_ids = list(link_map.keys())    # 获取所有 link ID
    # 遍历所有的 link ID，生成 OD 对
    for i in range(len(link_ids)):
        for j in range(i + 1, len(link_ids)):
            # 生成一个基础的 OD 对，默认值为 0
            od_pairs.append([link_ids[i], link_ids[j], 0])
    return od_pairs  # 返回所有生成的 OD 对

# 限制世界范围，设定可用的路网链接


class Sim:
    def __init__(self, xodr , show_plot=False):  # 初始化模拟类，接收相关参数
        # 如果需要绘图显示仿真过程
        if show_plot:
            self.fig, self.ax = plt.subplots()  # 创建一个新的图表和坐标轴
            plt.xlim(-30, 30)  # 设置x轴的显示范围
            plt.ylim(-30, 30)  # 设置y轴的显示范围
            plt.axis('equal')  # 保证坐标轴比例相等
        else:
            self.fig = None  # 如果不绘图，fig为None
            self.ax = None  # 如果不绘图，ax为None

        # 初始化xodr路网图
        roadD1 = D1xodrRoads.Graph()  # 创建一个xodr道路图的对象

        shp_flag = 0  # 标志，是否绘制路网形状（默认不绘制）

        # 从xodr文件读取道路信息，创建路网，并可选择性地绘制
        net = D1xodrRoads.create_road(roadD1, xodr, self.ax if show_plot else None)
        self.net = net  # 存储路网对象

        # 寻找起点和终点的连接（车辆OD）
        o_points = [link_id for link_id, link in net.link_map.items() if not link.in_link_lst]
        d_points = [link_id for link_id, link in net.link_map.items() if not link.out_link_lst]

        # 寻找合法的起点-终点连接
        legal_connections = find_legal_connections_between_o_d(net.link_map, o_points, d_points)

        # 组织合法的连接
        path_set = organize_legal_connections(legal_connections)

        # 如果没有合法的连接路径，则生成基础的OD对（起点-终点对）
        if not path_set:
            path_set = generate_basic_od_pairs(net.link_map)

        # 根据路网拓扑结构创建路径
        roadD1.create_path(path_set)

        # 初始化路径的时间间隔
        for path in roadD1.path_map.values():
            path.interval = 0

        # 约束路网，更新world链接
        self.constrain_world(roadD1)

        # 存储路网对象
        self.roadD1 = roadD1

        # 为所有路径创建路径ID列表
        all_path_list = []
        for path in self.roadD1.path_map.values():
            path_id_lst = self.roadD1.find_path(path.oid, path.did)  # 获取路径的ID列表
            all_path_list.append(path_id_lst)
        self.all_path_list = all_path_list  # 存储所有路径的ID列表

        # 更新起点和终点点的连接（完善路网）
        o_points = [link_id for link_id, link in net.link_map.items()]
        d_points = [link_id for link_id, link in net.link_map.items()]

        # 再次查找合法的连接
        legal_connections = find_legal_connections_between_o_d(net.link_map, o_points, d_points)

        # 组织合法的连接
        path_set = organize_legal_connections(legal_connections)

        # 如果路径为空，生成基础OD对
        if not path_set:
            path_set = generate_basic_od_pairs(net.link_map)

        # 初始化路径的时间间隔
        for path in roadD1.path_map.values():
            path.interval = 0

        # 约束世界（设置可用的道路ID列表）
        self.constrain_world(roadD1)

        # 存储路网数据
        self.roadD1 = roadD1


    def constrain_world(self,graph):
        world_link_ids = []  # 存储所有的世界路链ID
        for path in graph.path_map.values():
            roads = path.path_id_lst
            for road in roads:
                if road not in world_link_ids:
                    world_link_ids.append(road)
        graph.world = world_link_ids  # 设置图的世界范围（可用道路ID列表）


def extract_road_lane_coordinates(sim_obj):
    """提取道路和车道的坐标数据"""
    road_lane_coords = []

    # 遍历所有道路（link）
    for link_id, link in sim_obj.roadD1.link_map.items():
        # 获取当前link的上下游连接关系
        successor_ids = [out_link.id for out_link in link.out_link_lst]  # 后继link ID列表
        predecessor_ids = [in_link.id for in_link in link.in_link_lst]  # 前驱link ID列表

        road_data = {
            "road_id": link_id,
            "lanes": [],
            "road_successor": successor_ids,  # 添加后继link ID列表
            "road_predecessor": predecessor_ids  # 添加前驱link ID列表
        }

        # 遍历车道（lane）
        for lane in link.lane_lst:
            if hasattr(lane, 'xy') and lane.xy:
                # 转换坐标格式：[[x1,y1], [x2,y2], ...]
                coordinates = [[x, y] for x, y in zip(lane.xy[0], lane.xy[1])]

                lane_data = {
                    "lane_id": lane.id,
                    "coordinates": coordinates,
                    "type": lane.type,
                    "width": lane.width,
                    "global_lane_id": lane.index_id,  # 唯一ID格式：link_id*100 + lane_id
                    "successor": lane.out_lane_id_lst,  # 车道级后继
                    "predecessor": lane.in_lane_id_lst  # 车道级前驱
                }
                road_data["lanes"].append(lane_data)

        road_lane_coords.append(road_data)

    return road_lane_coords

def is_point_near(prediction, coordinates_data, threshold=1.8):
    """
    检测预测点是否接近路网中的任何车道点。

    参数:
        prediction (tuple): 目标点的坐标 (x, y)
        coordinates_data (list): 路网数据，包含各道路和车道的坐标信息
        threshold (float): 距离阈值，默认0.15米

    返回:
        bool: 存在邻近点返回True，否则返回False
    """
    x_pred, y_pred = prediction
    # 遍历每条道路
    for road in coordinates_data:
        # 遍历每个车道
        for lane in road['lanes']:
            # 遍历车道中的每个坐标点
            for coord in lane['coordinates']:
            # for coord in lane['center_line']:
                x, y = coord
                # 计算欧氏距离
                distance = ((x - x_pred)** 2 + (y - y_pred)** 2)** 0.5
                if distance < threshold:
                    return True
    return False


def get_nearest_road_lane(prediction, coordinates_data):
    """
    获取预测点最近的的道路和车道ID（无距离限制）

    参数:
        prediction (tuple): 目标点的坐标 (x, y)
        coordinates_data (list): 路网数据

    返回:
        dict: 包含 road_id, lane_id 和 distance 的字典
              如果没有数据则返回None
    """
    x_pred, y_pred = prediction
    min_distance = float('inf')
    closest_info = None

    for road in coordinates_data:
        road_id = road["road_id"]
        for lane in road["lanes"]:
            lane_id = lane["lane_id"]
            global_lane_id = lane["global_lane_id"]
            for coord in lane["coordinates"]:
                x, y = coord
                distance = math.hypot(x - x_pred, y - y_pred)
                if distance < min_distance:
                    min_distance = distance
                    closest_info = {
                        "road_id": road_id,
                        "lane_id": lane_id,
                        "global_lane_id": global_lane_id,
                        "distance": distance
                    }
    return closest_info




def get_car_in_lane(xodr_file,position):
    # xodr_file= '0_6_straight_straight_19.xodr'
    xodr = xml.dom.minidom.parse(xodr_file)

    xodr_end = Sim(xodr,show_plot=False)

    # 执行提取
    coordinates_data = extract_road_lane_coordinates(xodr_end)
    print(coordinates_data)

    # 判断预测值是否到地图边缘（判断地图中是否有临近点）
    # position = (1.5, 2.3)  # 替换为实际预测点
    nearest_road_lane = get_nearest_road_lane(position, coordinates_data)

    # 示例输出结构
    # print(f"共提取 {len(coordinates_data)} 条道路")
    # print(f"第一条道路含 {len(coordinates_data[0]['lanes'])} 条车道")
    # print(f"第一条车道坐标示例：{coordinates_data[0]['lanes'][0]['coordinates'][:2]}")
    # print(f"是否存在邻近点：{is_near}")
    return coordinates_data

xodr_file= '0_6_straight_straight_19.xodr'
position = (1.5, 2.3)
get_car_in_lane(xodr_file,position)