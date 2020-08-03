import csv
import os
import glob
import argparse

paser = argparse.ArgumentParser(description="make csv of for train.")
paser.add_argument('--path', type=str,
                   default='/home/du/disk2/Desk/dataset/ibox/cls/new_c15_525',
                   help='the path which include all image')
paser.add_argument('--output', type=str,
                   default='newc15_train.csv',
                   help='the path which save your csv')
args = paser.parse_args()


def main(args):
    path = args.path
    files = os.listdir(path)
    count = 0
    label_name = {}
    label_list = []
    print(files)
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i in files:
            if count not in label_name:
                label_name[count] = i
            if i not in label_list:
                label_list.append(i)

            class_path = os.path.join(path, i)
            for name in glob.glob(class_path+'/*'):
                print([name, count])
                writer.writerow([name, count])
            count += 1
    print('labels', label_name)
    print('labels_list', label_list)

def tbox_csv(args):
    path = args.path
    files = os.listdir(path)
    count = 0
    label_name = {}
    label_list = []
    print(files)
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i in files:
            if count not in label_name:
                label_name[count] = i
            if i not in label_list:
                label_list.append(i)

            class_path = os.path.join(path, i)
            # class_path = os.path.join(class_path, 'JPEGImages')
            for name in glob.glob(class_path+'/*'):
                print([name, count])
                writer.writerow([name, count])
            count += 1
    print('labels', label_name)
    print('labels_list', label_list)

    
if __name__ == '__main__':
    # main(args)
    tbox_csv(args)