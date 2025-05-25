import numpy as np
import pandas as pd
from math import atan2, degrees

# 常量定义
TIME_INTERVAL = 0.1  # 时间间隔(秒)
COMFORT_THRESHOLDS = {
    'lon_acc': [-9.8, 9.8],  # 纵向加速度 (动力学校核)
    'steer_angle': [-0.7, 0.7],  # 转向角 (rad)
    'lat_acc': [-0.5, 0.5],  # 横向加速度 (m/s²)
    'lon_jerk': [-6, 6],  # 纵向加加速度 (m/s³)
    'yaw_rate': [-0.5, 0.5]  # 横摆角速度 (rad/s)
}


def calculate_kinematics(df):
    """计算运动学参数"""
    # 按时间步排序
    df = df.sort_values(by='time_step')

    # 计算速度
    df['vx'] = df['x'].diff() / TIME_INTERVAL
    df['vy'] = df['y'].diff() / TIME_INTERVAL

    # 计算航向角（弧度）
    df['yaw'] = np.arctan2(df['vy'], df['vx'])

    # 计算横向加速度
    df['lat_acc'] = (df['yaw'].diff() / TIME_INTERVAL) * np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)

    # 计算纵向加加速度
    df['lon_jerk'] = df['lon_acc'].diff() / TIME_INTERVAL

    # 计算横摆角速度
    df['yaw_rate'] = df['yaw'].diff() / TIME_INTERVAL

    return df.dropna()


def check_dynamics(df):
    """执行动力学检查"""
    results = {
        '动力学校核': {'通过': True, '违规点': []},
        '舒适性指标': {
            '横向加速度超限': [],
            '纵向加加速度超限': [],
            '横摆角速度超限': []
        }
    }

    # 动力学校核
    mask = (df['lon_acc'] < COMFORT_THRESHOLDS['lon_acc'][0]) | \
           (df['lon_acc'] > COMFORT_THRESHOLDS['lon_acc'][1])
    if mask.any():
        results['动力学校核']['通过'] = False
        results['动力学校核']['违规点'] = df[mask]['time_step'].tolist()

    # 舒适性检查
    for ts, row in df.iterrows():
        # 横向加速度
        if not (COMFORT_THRESHOLDS['lat_acc'][0] <= row['lat_acc'] <= COMFORT_THRESHOLDS['lat_acc'][1]):
            results['舒适性指标']['横向加速度超限'].append(row['time_step'])

        # 纵向加加速度
        if not (COMFORT_THRESHOLDS['lon_jerk'][0] <= row['lon_jerk'] <= COMFORT_THRESHOLDS['lon_jerk'][1]):
            results['舒适性指标']['纵向加加速度超限'].append(row['time_step'])

        # 横摆角速度
        if not (COMFORT_THRESHOLDS['yaw_rate'][0] <= row['yaw_rate'] <= COMFORT_THRESHOLDS['yaw_rate'][1]):
            results['舒适性指标']['横摆角速度超限'].append(row['time_step'])

    return results


def generate_report(results, total_time):
    """生成验证报告"""
    report = []

    # 动力学校核结果
    dyn_check = results['动力学校核']
    report.append("=== 动力学校核 ===")
    report.append(f"结果: {'通过' if dyn_check['通过'] else '不通过'}")
    if not dyn_check['通过']:
        report.append(f"违规时间点: {dyn_check['违规点'][:5]}(共{len(dyn_check['违规点'])}处)")

    # 舒适性检查结果
    comfort = results['舒适性指标']
    report.append("\n=== 舒适性指标 ===")

    for metric in comfort:
        violation_count = len(comfort[metric])
        violation_ratio = violation_count / total_time
        report.append(
            f"{metric.replace('_', ' ')}: "
            f"超限次数 {violation_count} 次 "
            f"({violation_ratio:.1%})"
        )

    return "\n".join(report)


def main(file_path):
    # 读取数据
    columns = ['time_step', 'vehicle_id', 'x', 'y', 'lon_acc', 'steer_angle']
    raw_df = pd.read_csv(file_path, sep=' ', header=None, names=columns)

    # 按车辆分组处理
    final_report = []
    for vid, group in raw_df.groupby('vehicle_id'):
        # 计算运动学参数
        df = calculate_kinematics(group)

        # 执行检查
        results = check_dynamics(df)

        # 生成报告
        report = [
            f"\n车辆 {vid} 检查结果:",
            generate_report(results, len(df))
        ]
        final_report.extend(report)

    print("\n".join(final_report))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("使用方法: python validate_dynamics.py <轨迹文件路径>")
        sys.exit(1)

    main(sys.argv[1])