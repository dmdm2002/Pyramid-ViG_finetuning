import glob
import re
import pandas as pd
import numpy as np

AUG_IMAGE_FOLDER = f'C:/Users/rlawj/WORK/SIDE_PROJECT/DACON/BLOCK_CLASSIFICATION/DB/aug_train'
AUG_IMAGE_LIST = glob.glob(f'{AUG_IMAGE_FOLDER}/*')
# print(AUG_IMAGE_LIST)
csv_name = ['train', 'valid']

for file_name in csv_name:
    ORIGIANL_DB_INFO_PAHT = f'C:/Users/rlawj/WORK/SIDE_PROJECT/DACON/BLOCK_CLASSIFICATION/DB/DB_informations/{file_name}.csv'
    ORIGIANL_DB_INFO = pd.read_csv(ORIGIANL_DB_INFO_PAHT)
    ORIGIANL_DB_INFO_ARR = np.array(ORIGIANL_DB_INFO)

    aug_info_result_list = []
    for original_info in ORIGIANL_DB_INFO_ARR:
        for aug_info in AUG_IMAGE_LIST:
            aug_info_id = aug_info.split("\\")[-1]
            split_list = aug_info_id.split("__")
            aug_info_id = split_list[0]
            aug_background_id = split_list[1]
            if original_info[0] == aug_info_id:
                temp = original_info[2:].tolist()
                insert_id = f'{aug_info_id}__{aug_background_id}'
                image_path = f'C:/Users/rlawj/WORK/SIDE_PROJECT/DACON/BLOCK_CLASSIFICATION/random_choice_aug_image/{insert_id}.jpg'

                temp.insert(0, image_path)
                temp.insert(0, insert_id)
                aug_info_result_list.append(temp)

                continue

    print(f'FINISH {file_name}!!!')
    df = pd.DataFrame(aug_info_result_list, columns=['id', 'img_path', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    df.to_csv(f'C:/Users/rlawj/WORK/SIDE_PROJECT/DACON/BLOCK_CLASSIFICATION/DB/DB_informations/aug_{file_name}.csv', index=False)