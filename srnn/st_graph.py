import numpy as np
from helper import getMagnitudeAndDirection, getVector
from xodr.xodrpoints import get_car_in_lane

class ST_GRAPH:
    def __init__(self, batch_size):
        """
        Initializer function for the ST graph class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        """
        self.batch_size = batch_size
        self.nodes = [{} for i in range(batch_size)]
        self.edges = [{} for i in range(batch_size)]
        self.road_nodes = [{} for _ in range(batch_size)]  # 新增道路节点存储
        self.static_edges = [{} for _ in range(batch_size)]  # 新增静态边存储

    def reset(self):
        self.nodes = [{} for i in range(self.batch_size)]
        self.edges = [{} for i in range(self.batch_size)]
        self.road_nodes = [{} for _ in range(self.batch_size)]
        self.static_edges = [{} for _ in range(self.batch_size)]

    def readGraph(self, source_batch,seq_length, road_data_batch):
        """
        Main function that constructs the ST graph from the batch data
        params:
        source_batch : List of lists of numpy arrays. Each numpy array corresponds to a frame in the sequence.
        categories:  car --> 3,   2 --> bicycle ,  1 ---> pedestrian
        """
        self.seq_length = seq_length
        for sequence in range(self.batch_size):
            # source_seq is a list of numpy arrays
            # where each numpy array corresponds to a single frame
            source_seq = source_batch[sequence]
            road_data = road_data_batch[sequence]  # 获取道路数据

            # 新增数据检查
            print(f"Debug road_data type: {type(road_data)}, content:")
            for i, item in enumerate(road_data):
                print(f"Item {i}: type={type(item)}, content={str(item)[:50]}")  # 打印前50字符避免刷屏


            # 新增道路节点处理
            self._process_road_nodes(sequence, road_data)
            self._create_static_edges(sequence, road_data)
            #  list of frames, every frames may have different number person
            print('source_seq=',len(source_seq))
            print('seq_length=',self.seq_length)

            for framenum in range(self.seq_length):
                # Each frame is a numpy array
                # each row in the array is of the form
                # pedID, x, y, type
                frame = source_seq[
                    framenum
                ]

                # Add nodes
                for ped in range(frame.shape[0]):
                    pedID = frame[ped, 0]
                    x = frame[ped, 1]
                    y = frame[ped, 2]
                    # 将航向加进去
                    heading = frame[ped, 3]
                    pos = (x, y, heading)
                    node_type = frame[ped, 4]

                    if pedID not in self.nodes[sequence]:
                        node_id = pedID
                        node_pos_list = {}
                        node_pos_list[framenum] = pos
                        self.nodes[sequence][pedID] = ST_NODE(
                            node_type, node_id, node_pos_list
                        )
                    else:
                        self.nodes[sequence][pedID].addPosition(pos, framenum)
                        # Add Temporal edge between the node at current time-step
                        # and the node at previous time-step
                        edge_id = (pedID, pedID)
                        pos_edge = (
                            self.nodes[sequence][pedID].getPosition(framenum - 1),
                            # 只将x，y赋给边
                            pos[0:2],
                        )
                        if edge_id not in self.edges[sequence]:
                            if int(node_type) == 3:
                                edge_type = "car/T"
                            elif int(node_type) == 2:
                                edge_type = "bicycle/T"
                            elif int(node_type) == 1:
                                edge_type = "pedestrian/T"
                            else:
                                raise Exception("edge_type error")
                            edge_pos_list = {}
                            # ASSUMPTION: Adding temporal edge at the later time-step
                            edge_pos_list[framenum] = pos_edge
                            self.edges[sequence][edge_id] = ST_EDGE(
                                edge_type, edge_id, edge_pos_list
                            )
                        else:
                            self.edges[sequence][edge_id].addPosition(
                                pos_edge, framenum
                            )

                        # 新增车辆-道路边处理
                        self._create_road_edges(sequence, frame, framenum)

                # ASSUMPTION:
                # Adding spatial edges between all pairs of pedestrians.
                # TODO:
                # Can be pruned by considering pedestrians who are close to each other
                # Add spatial edges
                # all the spatial edges share same weights
                MAX_SPATIAL_DIST = 15.0  # 单位：米（根据场景调整）
                for ped_in in range(frame.shape[0]):
                    for ped_out in range(ped_in + 1, frame.shape[0]):

                        pos_in = (frame[ped_in, 1], frame[ped_in, 2])
                        pos_out = (frame[ped_out, 1], frame[ped_out, 2])

                        # 计算欧氏距离
                        dx = pos_in[0] - pos_out[0]
                        dy = pos_in[1] - pos_out[1]
                        distance = np.sqrt(dx ** 2 + dy ** 2)

                        if distance <= MAX_SPATIAL_DIST:  # 仅生成近距离边
                            # 原有创建边的逻辑
                            pedID_in = frame[ped_in, 0]
                            pedID_out = frame[ped_out, 0]
                            pos_in = (frame[ped_in, 1], frame[ped_in, 2])
                            pos_out = (frame[ped_out, 1], frame[ped_out, 2])
                            pos = (pos_in, pos_out)
                            edge_id = (pedID_in, pedID_out)
                            # ASSUMPTION:
                            # Assuming that pedIDs always are in increasing order in the input batch data
                            if edge_id not in self.edges[sequence]:
                                edge_type = "all_categories/S"
                                edge_pos_list = {}
                                edge_pos_list[framenum] = pos
                                self.edges[sequence][edge_id] = ST_EDGE(
                                    edge_type, edge_id, edge_pos_list
                                )
                            else:
                                self.edges[sequence][edge_id].addPosition(pos, framenum)

    def _process_road_nodes(self, sequence, road_data):
        """处理道路和车道节点"""
        if not isinstance(road_data, list):
            raise TypeError(f"road_data must be list, got {type(road_data)}")
        for road_link in road_data:  # 遍历每个road/link
            if not isinstance(road_link, dict):
                raise ValueError(f"Expected dict, got {type(road_link)}")
            # 创建道路层级的虚拟节点（表示整个link）
            road_node_id = f"road_{road_link['road_id']}"
            self.road_nodes[sequence][road_node_id] = ST_NODE(
                node_type=13,  # 新增类型：ROAD_TYPES['road'] = 13
                node_id=road_node_id,
                node_pos_list={t: (0, 0, 0) for t in range(self.seq_length)}  # 虚拟位置
            )

            # 处理车道数据
            for lane in road_link["lanes"]:
                coordinates = lane["coordinates"]
                # 每5米采样一个点（假设坐标单位为米）
                sampled_points = coordinates[::5]
                # 只保留起点和终点
                if len(sampled_points) > 2:
                    sampled_points = [coordinates[0], coordinates[-1]]
                # 保存采样后的坐标点
                lane["sampled_points"] = sampled_points
                # for point_idx, (x, y) in enumerate(lane["coordinates"]):
                for point_idx, (x, y) in enumerate(sampled_points):
                    lane_node_id = f"lane_{lane['global_lane_id']}_{point_idx}"
                    self.road_nodes[sequence][lane_node_id] = ST_NODE(
                        node_type=ST_NODE.ROAD_TYPES[lane['type']],  # 转换为数值类型
                        node_id=lane_node_id,
                        node_pos_list={t: (x, y, 0) for t in range(self.seq_length)}
                    )

    # 动态边创建
    def _create_road_edges(self, sequence, frame, framenum):
        """创建车辆与道路元素之间的边"""
        for ped in range(frame.shape[0]):
            pedID = frame[ped, 0]
            obj_type = int(frame[ped, 4])
            current_pos = (frame[ped, 1], frame[ped, 2])

            if obj_type == 3:  # 仅处理车辆
                # 寻找最近的车道线（示例逻辑）
                nearest_lane = self._find_nearest_road_element(
                    current_pos,
                    self.road_nodes[sequence],
                    elem_type='lane'
                )

                if nearest_lane:
                    # 创建车辆-车道线边
                    edge_id = (pedID, nearest_lane.node_id)
                    pos_edge = (current_pos, nearest_lane.getPosition(framenum)[:2])

                    if edge_id not in self.static_edges[sequence]:
                        self.static_edges[sequence][edge_id] = ST_EDGE(
                            edge_type="vehicle/lane_S",
                            edge_id=edge_id,
                            edge_pos_list={framenum: pos_edge}
                        )
                    else:
                        self.static_edges[sequence][edge_id].addPosition(pos_edge, framenum)

    def _find_nearest_road_element(self, current_pos, road_nodes, elem_type):
        """查找最近的指定类型道路元素（简化版）"""
        min_dist = float('inf')
        nearest_elem = None
        current_x, current_y = current_pos

        for node in road_nodes.values():
            if node.node_type == elem_type:
                node_x, node_y, _ = node.getPosition(0)  # 静态元素位置不随时间变化
                dist = np.sqrt((current_x - node_x)** 2 + (current_y - node_y)** 2)
                if dist < min_dist and dist < 5.0:  # 设置距离阈值
                    min_dist = dist
                    nearest_elem = node
        return nearest_elem

    def _create_static_edges(self, sequence, road_data):
        for road_link in road_data:
            # 道路级连接
            current_road_id = f"road_{road_link['road_id']}"
            for succ_road_id in road_link["road_successor"]:
                self._add_static_edge(
                    sequence,
                    current_road_id,
                    f"road_{succ_road_id}",
                    "road/road_S"
                )

            # 车道级连接
            for lane in road_link["lanes"]:
                lane_base_id = f"lane_{lane['global_lane_id']}"
                sampled_points = lane.get("sampled_points", lane["coordinates"])  # 获取采样后的点
                # 内部点连接
                for i in range(len(sampled_points) - 1):
                    self._add_static_edge(
                        sequence,
                        f"{lane_base_id}_{i}",
                        f"{lane_base_id}_{i + 1}",
                        "lane/lane_S"
                    )
                # 跨车道连接使用采样后的最后一个点
                current_end_idx = len(sampled_points) - 1
                # 跨车道连接
                for succ_lane_id in lane["successor"]:
                    self._add_static_edge(
                        sequence,
                        f"{lane_base_id}_{current_end_idx}",  # 当前车道末端
                        f"lane_{succ_lane_id}_0",  # 后继车道起点
                        "lane/lane_S"
                    )
    def _add_static_edge(self, sequence, from_id, to_id, edge_type):
        """辅助函数：添加双向静态边"""
        edge_id = (from_id, to_id)
        reverse_edge_id = (to_id, from_id)

        # 正向边
        self.static_edges[sequence][edge_id] = ST_EDGE(
            edge_type=edge_type,
            edge_id=edge_id,
            edge_pos_list={t: ((0, 0), (0, 0)) for t in range(self.seq_length)}  # 静态边不需要位置变化
        )

        # 反向边（若需要）
        if "lane/lane_S" in edge_type:
            self.static_edges[sequence][reverse_edge_id] = ST_EDGE(
                edge_type=edge_type,
                edge_id=reverse_edge_id,
                edge_pos_list={t: ((0, 0), (0, 0)) for t in range(self.seq_length)}
            )




    def printGraph(self):
        """
        Print function for the graph
        For debugging purposes
        """
        for sequence in range(self.batch_size):
            nodes = self.nodes[sequence]
            edges = self.edges[sequence]

            print("Printing Nodes")
            print("===============================")
            for node in nodes.values():
                node.printNode()
                print("--------------")

            print("Printing Edges")
            print("===============================")
            for edge in edges.values():
                edge.printEdge()
                print("--------------")

    def getSequence(self,seq_length):
        """
        Gets the sequence
        """
        self.seq_length = seq_length
#########################################################################################
        # nodes = self.nodes[0]
        # edges = self.edges[0]
        nodes = {**self.nodes[0], **self.road_nodes[0]}  # 合并移动节点和道路节点
        edges = {**self.edges[0], ** self.static_edges[0]}  # 合并动态边和静态边
#########################################################################################

        numNodes = len(nodes.keys())
        # print("********************* numNodes {}***********".format(numNodes))
        list_of_nodes = {}

        retNodes = np.zeros((self.seq_length, numNodes, 3), dtype=np.float32)
        retEdges = np.zeros((self.seq_length, numNodes * numNodes, 2), dtype=np.float16)  # Diagonal contains temporal edges
        retNodePresent = [[] for c in range(self.seq_length)]
        retEdgePresent = [[] for c in range(self.seq_length)]

        # retNodes_type = [[] for c in range(self.seq_length)]
        # retEdges_type = [[] for c in range(self.seq_length)]

        for i, ped in enumerate(nodes.keys()):
            list_of_nodes[ped] = i
            pos_list = nodes[ped].node_pos_list
            for framenum in range(self.seq_length):
                if framenum in pos_list:
                    retNodePresent[framenum].append((i, nodes[ped].getType()))
                    retNodes[framenum, i, :] = list(pos_list[framenum])
                    # retNodes_type[framenum].append(nodes[ped].getType())

        for ped, ped_other in edges.keys():
            i, j = list_of_nodes[ped], list_of_nodes[ped_other]
            edge = edges[(ped, ped_other)]
            if ped == ped_other:
                # Temporal edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((i, j, edge.getType()))
                        retEdges[framenum, i * (numNodes) + j, :] = getVector(
                            edge.edge_pos_list[framenum]
                        )  # Gets the vector pointing from second element to first element
            else:
                # Spatial edge
                for framenum in range(self.seq_length):
                    if framenum in edge.edge_pos_list:
                        retEdgePresent[framenum].append((i, j, edge.getType()))
                        retEdgePresent[framenum].append((j, i, edge.getType()))
                        # the position returned is a tuple of tuples

                        retEdges[framenum, i * numNodes + j, :] = getVector(
                            edge.edge_pos_list[framenum]
                        )
                        retEdges[framenum, j * numNodes + i, :] = -np.copy(
                            retEdges[framenum, i * (numNodes) + j, :]
                        )

        return retNodes, retEdges, retNodePresent, retEdgePresent


class ST_NODE:
    # 添加新的道路类型处理
    ROAD_TYPES = {
        'lane': 10,
        'road_edge': 11,
        'traffic_light': 12,
        'road': 13, # 新增道路层级类型

        'driving': 20,
        'special1':21,
        'offRamp':22,
        'onRamp':23

    }
    def __init__(self, node_type, node_id, node_pos_list):
        """
        Initializer function for the ST node class
        params:
        node_type : Type of the node (Human or Obstacle)
        node_id : Pedestrian ID or the obstacle ID
        node_pos_list : Positions of the entity associated with the node in the sequence
        """
#####################################################################################
        if isinstance(node_type, str):  # 支持字符串类型
            self.node_type = self.ROAD_TYPES.get(node_type, 0)
        else:
            self.node_type = node_type
#####################################################################################
        self.node_type = node_type
        self.node_id = node_id
        self.node_pos_list = node_pos_list

    def getPosition(self, index_i):
        """
        Get the position of the node at time-step index in the sequence
        params:
        index : time-step
        """
        if index_i not in self.node_pos_list:
            tmp_list = sorted(list(self.node_pos_list.keys()))
            last_index = [i for i in tmp_list if i < index_i][-1]
            return self.node_pos_list[last_index]
        # assert index in self.node_pos_list
        return self.node_pos_list[index_i]

    def getType(self):
        """
        Get node type
        """
        return self.node_type

    def getID(self):
        """
        Get node ID
        """
        return self.node_id

    def addPosition(self, pos, index):
        """
        Add position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        """
        assert index not in self.node_pos_list
        self.node_pos_list[index] = pos

    def printNode(self):
        """
        Print function for the node
        For debugging purposes
        """
        print(
            "Node type:",
            self.node_type,
            "with ID:",
            self.node_id,
            "with positions:",
            self.node_pos_list.values(),
            "at time-steps:",
            self.node_pos_list.keys(),
        )


class ST_EDGE:
#########################################################################################
    EDGE_TYPES = {
        'vehicle/lane_S': 20,
        'road/road_S': 21,  # 道路层级的连接
        'lane/lane_S': 22  # 车道点之间的连接
    }
#########################################################################################
    def __init__(self, edge_type, edge_id, edge_pos_list):
        """
        Inititalizer function for the ST edge class
        params:
        edge_type : Type of the edge (Human-Human or Human-Obstacle)
        edge_id : Tuple (or set) of node IDs involved with the edge
        edge_pos_list : Positions of the nodes involved with the edge
        """

#########################################################################################
        if isinstance(edge_type, str):
            self.edge_type = self.EDGE_TYPES.get(edge_type, 0)
        else:
            self.edge_type = edge_type
#########################################################################################

        self.edge_type = edge_type
        self.edge_id = edge_id
        self.edge_pos_list = edge_pos_list

    def getPositions(self, index):
        """
        Get Positions of the nodes at time-step index in the sequence
        params:
        index : time-step
        """
        assert index in self.edge_pos_list
        return self.edge_pos_list[index]

    def getType(self):
        """
        Get edge type
        """
        return self.edge_type

    def getID(self):
        """
        Get edge ID
        """
        return self.edge_id

    def addPosition(self, pos, index):
        """
        Add a position to the pos_list at a specific time-step
        params:
        pos : A tuple (x, y)
        index : time-step
        """
        assert index not in self.edge_pos_list
        self.edge_pos_list[index] = pos

    def printEdge(self):
        """
        Print function for the edge
        For debugging purposes
        """
        # print('Edge type:', self.edge_type, 'between nodes:', self.edge_id, 'at time-steps:', self.edge_pos_list.keys())
        print(
            "Edge type:",
            self.edge_type,
            "between nodes:",
            self.edge_id,
            "with positions:",
            self.edge_pos_list.values(),
            "at time-steps:",
            self.edge_pos_list.keys(),
        )
