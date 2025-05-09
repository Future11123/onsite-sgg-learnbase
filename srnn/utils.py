import logging
import os
import pickle
import random
import glob
import numpy as np


import sys

# 获取当前文件的绝对路径，并定位到项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 然后继续其他导入
from xodr.xodrpoints import get_car_in_lane

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


class DataLoader:
    def __init__(
        self, batch_size=1, obs_length=31, forcePreProcess=True, infer=False
    ):
        """
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered  21
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        """
        # self.min_position_x = float('inf')
        # self.max_position_x = -float('inf')
        # self.min_position_y = float('inf')
        # self.max_position_y = -float('inf')
        # self.min_heading = float('inf')
        # self.max_heading = -float('inf')

        self.min_position_x = []
        self.max_position_x = []
        self.min_position_y = []
        self.max_position_y = []
        self.min_heading = []
        self.max_heading = []

        # random.seed(42)
        # np.random.seed(42)
        # List of data directories where raw data resides
        # self.data_dirs = "../data/prediction_train/"
        # self.dataset_cnt = len(os.listdir(self.data_dirs))
        # self.dataset_idx = sorted(os.listdir(self.data_dirs))

        self.base_dir = "../第五赛道_A卷/"
        # 获取所有包含_gt.txt的子目录路径（确保路径正确）
        self.dataset_dirs = []
        for entry in os.listdir(self.base_dir):
            entry_path = os.path.join(self.base_dir, entry)
            if os.path.isdir(entry_path):
                gt_files = glob.glob(os.path.join(entry_path, "*_gt.txt"))
                if gt_files:
                    self.dataset_dirs.append(entry_path)
        self.dataset_dirs = sorted(self.dataset_dirs)
        self.dataset_cnt = len(self.dataset_dirs)
        self.dataset_idx = self.dataset_dirs
        np.random.shuffle(self.dataset_idx)  # 打乱顺序
        self.train_data_dirs = self.dataset_idx

        if infer == True:
            self.train_data_dirs = self.dataset_idx[int(self.dataset_cnt * 0.9) :]
        self.infer = infer

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = []
        self.obs_length = obs_length

        # data_file = os.path.join("../data/", "trajectories.cpkl")
        data_file = os.path.join("../第五赛道_A卷/", "trajectories.cpkl")
        if infer == True:
            # data_file = os.path.join("../data/", "test_trajectories.cpkl")
            data_file = os.path.join("../第五赛道_A卷/", "test_trajectories.cpkl")

        self.val_fraction = 0.2

        # If the file doesn't exist or forcePreProcess is true
        if not (os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.train_data_dirs, data_file)
            for data_index in range(len(self.dataset_idx)):
                print("预处理完成，归一化参数:", self.get_normalization_params(data_index))

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

        # # 初始化边界值
        # self.min_position_x = None
        # self.max_position_x = None
        # self.min_position_y = None
        # self.max_position_y = None
        # self.max_heading    = None
        # self.min_heading    = None



    def get_normalization_params(self,data_index):
        """获取全局归一化参数"""
        return {
            'position': {
                'x': (self.min_position_x[data_index], self.max_position_x[data_index]),
                'y': (self.min_position_y[data_index], self.max_position_y[data_index])
            },
            'heading': (self.min_heading[data_index], self.max_heading[data_index])
        }


    def class_objtype(self, object_type):
        if object_type != 10:
            return 3
        else:
            return -1

    def frame_preprocess(self, data_dirs, data_file):
        """
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        """
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []
        # Validation frame data
        # valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        xodr_data = []  # 新增存储xodr内容的列表
        dataset_index = 0

        min_position_x = 15000
        max_position_x = -15000
        min_position_y = 15000
        max_position_y = -15000
        min_heading    = 6.5
        max_heading    = -6.5

        for ind_directory, directory in enumerate(data_dirs):
            # file_path = os.path.join("../data/prediction_train/", directory)
            gt_path = glob.glob(os.path.join(directory, "*_gt.txt"))[0]
            base_name = os.path.basename(gt_path).replace("_gt.txt", "")
            xodr_path = os.path.join(directory, f"{base_name}.xodr")

            if not os.path.exists(xodr_path):
                raise FileNotFoundError(f"Missing xodr file for {gt_path}")

            # 读取并存储xodr数据
            road_data_batch = get_car_in_lane(xodr_path)
            xodr_data.append(road_data_batch)

            data = np.genfromtxt(gt_path, delimiter=" ")

            # 检查数据有效性
            if data.size == 0 or data.shape[0] == 0:
                raise ValueError(f"文件 {gt_path} 为空或格式错误")

            # 使用numpy的min/max代替Python内置函数
            min_position_x = min(min_position_x, np.min(data[:, 2]))
            max_position_x = max(max_position_x, np.max(data[:, 2]))
            min_position_y = min(min_position_y, np.min(data[:, 3]))
            max_position_y = max(max_position_y, np.max(data[:, 3]))
            min_heading = min(min_heading, np.min(data[:, 4]))
            max_heading = max(max_heading, np.max(data[:, 4]))


            # min_position_x = min(min_position_x, min(data[:, 2]))
            # max_position_x = max(max_position_x, max(data[:, 2]))
            # min_position_y = min(min_position_y, min(data[:, 3]))
            # max_position_y = max(max_position_y, max(data[:, 3]))
            # min_heading = min(min_heading, min(data[:, 4]))
            # max_heading = max(max_heading, max(data[:, 4]))

            # # 保存边界值到类属性
            self.min_position_x.append(min_position_x)
            self.max_position_x.append(max_position_x)
            self.min_position_y.append(min_position_y)
            self.max_position_y.append(max_position_y)
            self.max_heading.append(max_heading)
            self.min_heading.append(min_heading)

        if None in [self.min_position_x, self.max_position_x,
                    self.min_position_y, self.max_position_y,
                    self.min_heading, self.max_heading]:
            raise ValueError("归一化参数未正确计算!")


        # For each dataset
        for ind_directory, directory in enumerate(data_dirs):
            # define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')
            file_path = glob.glob(os.path.join(directory, "*_gt.txt"))[0]
            # file_path = os.path.join("../data/prediction_train/", directory)

            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=" ")
            """
            data[:, 3] = (
                (data[:, 3] - min(data[:, 3])) / (max(data[:, 3]) - min(data[:, 3]))
            ) * 2 - 1
            data[:, 4] = (
                (data[:, 4] - min(data[:, 4])) / (max(data[:, 4]) - min(data[:, 4]))
            ) * 2 - 1
            """
            data[:, 2] = (
                (data[:, 2] - self.min_position_x[ind_directory]) / (self.max_position_x[ind_directory] - self.min_position_x[ind_directory])
            ) * 2 - 1
            data[:, 3] = (
                (data[:, 3] - self.min_position_y[ind_directory]) / (self.max_position_y[ind_directory] - self.min_position_y[ind_directory])
            ) * 2 - 1
            # 新增对航向的归一化
            data[:, 4] = (
                (data[:, 4] - self.min_heading[ind_directory]) / (self.max_heading[ind_directory] - self.min_heading[ind_directory])
            ) * 2 - 1

            # data = data[~(data[:, 2] == 5)]

            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[:, 0]).tolist()
            numFrames = len(frameList)

            # 将序列长度动态设为该数据集的帧数
            self.seq_length.append(numFrames)  # 记录当前数据集的 seq_length

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            # valid_frame_data.append([])

            skip = 1

            for ind, frame in enumerate(frameList):

                ## NOTE CHANGE
                if ind % skip != 0:
                    # Skip every n frames
                    continue

                # Extract all pedestrians in current frame
                pedsInFrame = data[data[:, 0] == frame, :]

                # Extract peds list
                pedsList = pedsInFrame[:, 1].tolist()

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []
                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[pedsInFrame[:, 1] == ped, 2][0]
                    current_y = pedsInFrame[pedsInFrame[:, 1] == ped, 3][0]
                    current_heading = pedsInFrame[pedsInFrame[:, 1] == ped, 4][0]
                    current_type = self.class_objtype(
                        int(pedsInFrame[pedsInFrame[:, 1] == ped, 5][0]))
                    # print('current_type    {}'.format(current_type))
                    # Add their pedID, x, y, heading to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y, current_heading, current_type])

                # if (ind > numFrames * self.val_fraction) or self.infer:
                #     # At inference time, no validation data
                #     # Add the details of all the peds in the current frame to all_frame_data
                #     all_frame_data[dataset_index].append(
                #         np.array(pedsWithPos)
                #     )  # different frame (may) have different number person
                # else:
                #     valid_frame_data[dataset_index].append(np.array(pedsWithPos))
                # 删除验证集
                all_frame_data[dataset_index].append( np.array(pedsWithPos))


            dataset_index += 1
        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump(
            (all_frame_data, frameList_data, numPeds_data,xodr_data),
            f,
            protocol=2,
        )
        f.close()

    def load_preprocessed(self, data_file):
        """
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        """
        # Load data from the pickled file
        f = open(data_file, "rb")
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.xodr_data = self.raw_data[3]  # 加载xodr数据
        # self.valid_data = self.raw_data[3]
        counter = 0
        valid_counter = 0

        print(len(self.data))


        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            # valid_frame_data = self.valid_data[dataset]
            print(
                "Training data from dataset {} : {}".format(
                    dataset, len(all_frame_data)
                )
            )
            # print(
            #     "Validation data from dataset {} : {}".format(
            #         dataset, len(valid_frame_data)
            #     )
            # )
            # Increment the counter with the number of sequences in the current dataset
            counter += 1


        # Calculate the number of batches
        self.num_batches = int(counter / self.batch_size)
        # self.valid_num_batches = int(valid_counter / self.batch_size)
        print("Total number of training batches: {}".format(self.num_batches * 2))
        # print("Total number of validation batches: {}".format(self.valid_num_batches))
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2
        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self, randomUpdate=False):
        """
        Function to get the next batch of points
        """
        # Source data
        x_batch = []
        # Target data 没用
        y_batch = []
        # Frame data
        frame_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0



        while i < self.batch_size:
            # Extract the frame data of the current dataset

            frame_data = self.data[self.dataset_pointer]
            frame_ids = self.frameList[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            seq_length = len(frame_data)


            if idx + seq_length <= len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx : idx + seq_length]

                # seq_target_frame_data = frame_data[idx + 1 : idx + self.obs_length + 1]
                seq_frame_ids = frame_ids[idx : idx + seq_length]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                # y_batch.append(seq_target_frame_data)
                frame_batch.append(seq_frame_ids)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, seq_length)
                else:
                    self.frame_pointer += seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)


        return x_batch, frame_batch, d

    # def next_valid_batch(self):
    #     """
    #     Function to get the next Validation batch of points
    #     """
    #     # Source data
    #     x_batch = []
    #     # Target data
    #     # y_batch = []
    #     # Dataset data
    #     d = []
    #     # Iteration index
    #     i = 0
    #     while i < self.batch_size:
    #         # Extract the frame data of the current dataset
    #         frame_data = self.valid_data[self.valid_dataset_pointer]
    #         # Get the frame pointer for the current dataset
    #         idx = self.valid_frame_pointer
    #         # While there is still seq_length number of frames left in the current dataset
    #         if idx + self.seq_length+1 <= len(frame_data):
    #             # All the data in this sequence
    #             # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
    #             seq_source_frame_data = frame_data[idx : idx + self.seq_length+1]
    #             # seq_target_frame_data = frame_data[idx + 1 : idx + self.obs_length + 1]
    #
    #             # Number of unique peds in this sequence of frames
    #             x_batch.append(seq_source_frame_data)
    #             # y_batch.append(seq_target_frame_data)
    #
    #             # advance the frame pointer to a random point
    #             # if randomUpdate:
    #             #     self.valid_frame_pointer += random.randint(1, self.obs_length)
    #             # else:
    #             #     self.valid_frame_pointer += 1
    #             #
    #             # d.append(self.valid_dataset_pointer)
    #             # i += 1
    #             self.frame_pointer += 1
    #             i += 1
    #
    #         else:
    #             # Not enough frames left
    #             # Increment the dataset pointer and set the frame_pointer to zero
    #             self.tick_batch_pointer(valid=True)
    #
    #     return x_batch, d

    def tick_batch_pointer(self, valid=False):
        """
        Advance the dataset pointer
        """
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self, valid=False):
        """
        Reset all pointers
        """
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
