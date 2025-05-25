import numpy as np
import pickle
import os


def parse_pkl_to_dict(pkl_path):
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)
    return data


def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            input_pkl = os.path.join(input_folder, filename)
            output_txt = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')

            data = parse_pkl_to_dict(input_pkl)

            if 'Ego' in data['ids']:
                ego_index = data['ids'].index('Ego')
                filtered_ids = [id for id in data['ids'] if id != 'Ego']
                filtered_positions = np.delete(data['positions'], ego_index, axis=0)
                filtered_headings = np.delete(data['headings'], ego_index, axis=0)
                filtered_valid_mask = np.delete(data['valid_mask'], ego_index, axis=0)
                filtered_types = np.delete(data['types'], ego_index)
            else:
                filtered_ids = data['ids']
                filtered_positions = data['positions']
                filtered_headings = data['headings']
                filtered_valid_mask = data['valid_mask']
                filtered_types = data['types']

            with open(output_txt, 'w') as txtfile:
                txtfile.write('frame,id,x,y,heading,type\n')

                for idx, obj_id in enumerate(filtered_ids):
                    for frame in range(filtered_valid_mask.shape[1]):
                        if filtered_valid_mask[idx, frame]:
                            x, y = filtered_positions[idx, frame]
                            heading = filtered_headings[idx, frame]
                            obj_type = filtered_types[idx]
                            line = f'{frame + 1},{obj_id},{x},{y},{heading},{obj_type}\n'
                            txtfile.write(line)


if __name__ == '__main__':
    input_folder = r"D:\Github\Onsite_rule_driven_model-main\sample\scene\mixed_952_32_1"
    output_folder = r'D:\Desktop\output'
    main(input_folder, output_folder)
    