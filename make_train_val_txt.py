"""
Make json files for dataset
"""
import json
import os

def get_val(root):

    with open("./val.json") as f:
        val_list = json.load(f)
    new_val = []
    for item in val_list:
        new_item = os.path.basename(item)
        new_val.append(new_item)
    fval = open('val.txt', 'w')
    for filename in new_val:
        fval.write(filename + '\n')
    fval.close()

def get_train(root):
    path = os.path.join(root, 'train_data', 'images')
    filenames = os.listdir(path)
    ftrain = open('train.txt', 'w')
    for filename in filenames:
        ftrain.write(filename + '\n')
    ftrain.close()

def get_test(root):
    path = os.path.join(root, 'test_data', 'images')
    filenames = os.listdir(path)
    ftest = open('test.txt', 'w')
    for filename in filenames:
        ftest.write(filename + '\n')
    ftest.close()

if __name__ == '__main__':
    root = os.path.join(os.getcwd(), 'original-data/')  # Dataset path
    get_train(root)
    get_val(root)
    print('Finish!')

