from visualization import visualize_sg_map
import pickle

if __name__ == '__main__':
    sg_path = './transferset/DIOR/00171.pickle'
    with open(sg_path,'rb') as f:
        sg = pickle.load(f)
    print(sg.shape)
    visualize_sg_map(sg)
    