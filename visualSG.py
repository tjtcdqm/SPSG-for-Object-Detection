from visualization import visualize_sg_map
import pickle

if __name__ == '__main__':
    sg_path = './transferset/DIOR/13319.pickle'
    with open(sg_path,'rb') as f:
        sg = pickle.load(f)
    visualize_sg_map(sg)
    