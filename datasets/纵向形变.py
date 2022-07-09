import os
import xml.etree.ElementTree as ET

def mkdir(path):
    '''
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    '''
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

if __name__ == '__main__':
    root_path = r"F:\Dataset\DiBei\preprocess_0410_train_aug"
    save_path = r"F:\Dataset\DiBei\preprocess_0410_train_aug_only1"
    mkdir(save_path)
    xml_filenames = os.listdir(root_path)
    for xml_filename in xml_filenames:
        if xml_filename.endswith("xml"):
            doc = ET.parse(root_path +'\\' + xml_filename)
            root = doc.getroot()
            for node in root:
                if node.tag == 'object':
                    if node[0].text == 'center_segregationC' or node[0].text == 'center_segregationB' or node[0].text == 'center_segregationA' or node[0].text == 'center_porosity' or node[0].text == 'center_crack' or node[0].text == 'center_segregation' :
                        node[0].text = 'target'
            doc.write(save_path + '\\' + xml_filename,encoding='utf-8')
            # 保存修改
            print('修改成功！')