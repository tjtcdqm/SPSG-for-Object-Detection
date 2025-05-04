from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(paths=['./datasets/DIOR'],
                    val_paths=[],
                      test_paths=['./datasets/DIOR'],
                      output_folder='./datasets/stealing_NWPU-VHR')