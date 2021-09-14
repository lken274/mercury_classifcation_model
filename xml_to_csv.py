import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import random


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text

            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     label,
                     xmin,
                     ymin,
                     xmax,
                     ymax
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'dataset/images/annotations')
    xml_df = xml_to_csv(image_path)
    #split into 98% train, 2% test
    split_index = int(len(xml_df) * 0.98)

    xml_train = xml_df[0:split_index]
    xml_test = xml_df[split_index:]
    xml_train.to_csv('labels_train.csv', index=None)
    xml_test.to_csv('labels_test.csv', index=None)
    
    print('Successfully converted xml to csv.')


main()

