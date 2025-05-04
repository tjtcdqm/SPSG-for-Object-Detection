import os
import torch
import os.path as osp
import xml.etree.ElementTree as ET
import json
# Label map
voc_labels = ('airplane','ship','storagetank','baseballfield','tenniscourt',
              'basketballcourt','groundtrackfield','harbor','bridge','vehicle')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
label_object = {k: 0 for _, k in enumerate(voc_labels)}
def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        # difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            # print(label)
            continue
        label_object[label]+=1
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        # difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels}
def create_data_lists(paths,val_paths,test_paths, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param paths: a list of paths contain voc data, merge their trainval together for training
    :param test_path: folder where the use its test.txt to test
    :param output_folder: folder where the JSONs must be saved
    """

    for i, path in enumerate(paths):
        paths[i] = os.path.abspath(path)

    for i, path in enumerate(val_paths):
        val_paths[i] = os.path.abspath(path)

    for i, path in enumerate(test_paths):
        test_paths[i] = os.path.abspath(path)
    # voc07_path = os.path.abspath(voc07_path)
    # voc12_path = os.path.abspath(voc12_path)

    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in paths:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))

    # valid data
    val_images = list()
    val_objects = list()
    n_objects = 0
    for val_path in val_paths:

        # Find IDs of images in the val data
        with open(os.path.join(val_path, 'ImageSets/Main/val.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(val_path, 'Annotations', id + '.xml'))
            if len(objects) == 0:
                continue
            if len(objects['boxes']) == 0:
                # print("val ",id)
                continue
            val_objects.append(objects)
            n_objects += len(objects)
            val_images.append(os.path.join(val_path, 'JPEGImages', id + '.jpg'))

        assert len(val_objects) == len(val_images)

    # Save to file
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(val_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(val_objects, j)

    print('\nThere are %d val images containing a total of %d objects. Files have been saved to %s.' % (
        len(val_images), n_objects, os.path.abspath(output_folder)))
    
    # Test data
    test_images = list()
    test_objects = list()
    n_objects = 0
    for test_path in test_paths:

        # Find IDs of images in the test data
        with open(os.path.join(test_path, 'ImageSets/Main/test.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(test_path, 'Annotations', id + '.xml'))
            if len(objects) == 0:
                continue
            if len(objects['boxes']) == 0:
                # print("val ",id)
                continue
            test_objects.append(objects)
            n_objects += len(objects)
            test_images.append(os.path.join(test_path, 'JPEGImages', id + '.jpg'))

        assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d test images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))

    print(label_object)
def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # print(set_1.shape,set_2.shape)
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    del lower_bounds, upper_bounds
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    with torch.no_grad():
        # Find intersections
        intersection = find_intersection(set_1, set_2)  # (n1, n2)

        # Find areas of each box in both sets
        areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
        areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

        # Find the union
        # PyTorch auto-broadcasts singleton dimensions
        union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
        iou = intersection / union
        del intersection, union, areas_set_1, areas_set_2
    # torch.cuda.empty_cache()
    return iou  # (n1, n2)