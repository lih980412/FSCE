import numpy as np
import pandas as pd
import shutil
import json
import os
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.font_manager import FontProperties
from PIL import Image
import random
import collections


myfont = FontProperties(fname=r"SourceHanSansSC-Bold.otf", size=20)
plt.rcParams['figure.figsize'] = (30, 30)
plt.rcParams['font.family']= myfont.get_family()
plt.rcParams['font.sans-serif'] = myfont.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"font.size":20})






def generate_anno_eda(dataset_path, anno_file):
    with open(os.path.join(dataset_path, anno_file), encoding='utf-8') as f:
        anno = json.load(f)
    print('标签类别:', anno['categories'])
    print('类别数量：', len(anno['categories']))
    print('训练集图片数量：', len(anno['images']))
    print('训练集标签数量：', len(anno['annotations']))

    total = []
    for img in anno['images']:
        hw = (img['height'], img['width'])
        total.append(hw)
    unique = set(total)
    for k in unique:
        print('长宽为(%d,%d)的图片数量为：' % k, total.count(k))

    ids = []
    images_id = []
    for i in anno['annotations']:
        ids.append(i['id'])
        images_id.append(i['image_id'])
    print('训练集图片数量:', len(anno['images']))
    print('unique id 数量：', len(set(ids)))
    print('unique image_id 数量', len(set(images_id)))

    # 创建类别标签字典
    category_dic = dict([(i['id'], i['name']) for i in anno['categories']])
    counts_label = dict([(i['name'], 0) for i in anno['categories']])
    for i in anno['annotations']:
        counts_label[category_dic[i['category_id']]] += 1
    label_list = counts_label.keys()  # 各部分标签
    print('标签列表:', label_list)
    size = counts_label.values()  # 各部分大小
    color = ['#FFB6C1', '#D8BFD8', '#9400D3', '#483D8B', '#4169E1', '#00FFFF', '#B1FFF0', '#ADFF2F', '#EEE8AA',
             '#FFA500', '#FF6347']  # 各部分颜色
    # explode = [0.05, 0, 0]   # 各部分突出值
    patches, l_text, p_text = plt.pie(size, labels=label_list, colors=color, labeldistance=1.1, autopct="%1.1f%%",
                                      shadow=False, startangle=90, pctdistance=0.6)
                                      # textprops={'fontproperties': myfont})
    plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
    plt.legend(prop=myfont)
    plt.show()

def get_all_bboxes(df, name):
    image_bboxes = df[df.file_name == name]

    bboxes = []
    categories = []
    for _, row in image_bboxes.iterrows():
        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_w, row.bbox_h, row.category_id))
    return bboxes

def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    color = ['#FFB6C1', '#D8BFD8', '#9400D3', '#483D8B', '#4169E1', '#00FFFF', '#B1FFF0', '#ADFF2F', '#EEE8AA',
             '#FFA500', '#FF6347']  # 各部分颜色
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            name = df.iloc[idx]["file_name"]
            img = Image.open(TRAIN_DIR + str(name))
            axs[row, col].imshow(img)

            bboxes = get_all_bboxes(df, name)
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                                         edgecolor=color[bbox[4]], facecolor='none')
                axs[row, col].add_patch(rect)

            axs[row, col].axis('off')

    plt.suptitle(title, fontproperties=myfont)

def plot_gray_examples(df, rows=3, cols=3, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
    color = ['#FFB6C1', '#D8BFD8', '#9400D3', '#483D8B', '#4169E1', '#00FFFF', '#B1FFF0', '#ADFF2F', '#EEE8AA',
             '#FFA500', '#FF6347']  # 各部分颜色
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            name = df.iloc[idx]["file_name"]
            img = Image.open(TRAIN_DIR + str(name)).convert('L')
            axs[row, col].imshow(img)

            bboxes = get_all_bboxes(df, name)
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                                         edgecolor=color[bbox[4]], facecolor='none')
                axs[row, col].add_patch(rect)

            axs[row, col].axis('off')

    plt.suptitle(title, fontproperties=myfont)

def get_image_brightness(image):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # get average brightness
    return np.array(gray).mean()

def add_brightness(df):
    brightness = []
    for _, row in df.iterrows():
        name = row["file_name"]
        image = cv2.imread(TRAIN_DIR + name)
        brightness.append(get_image_brightness(image))

    brightness_df = pd.DataFrame(brightness)
    brightness_df.columns = ['brightness']
    df = pd.concat([df, brightness_df], ignore_index=True, axis=1)
    df.columns = ['file_name', 'brightness']

    return df


# 计算IOU
def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# tmp 是一个pandas Series，且索引从0开始
def bbox_iou(tmp):
    iou_agg = 0
    iou_cnt = 0
    for i in range(len(tmp)):
        for j in range(len(tmp)):
            if i != j:
                iou_agg += bb_intersection_over_union(tmp[i], tmp[j])
                if bb_intersection_over_union(tmp[i], tmp[j]) > 0:
                    iou_cnt += 1
    iou_agg = iou_agg/2
    iou_cnt = iou_cnt/2
    return iou_agg, iou_cnt


# 计算bbox的RGB
def bb_rgb_cal(img, boxA):
    boxA = [int(x) for x in boxA]
    boxA = [boxA[0], boxA[1], boxA[0]+boxA[2], boxA[1]+boxA[3]]
    img = img.crop(boxA)
    width = img.size[0]
    height = img.size[1]
    img = img.convert('RGB')
    array = []
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x,y))
            rgb = (r, g, b)
            array.append(rgb)
    return round(np.mean(array[0]),2), round(np.mean(array[1]),2), round(np.mean(array[2]),2)

if __name__ == "__main__":
    # Setup the paths to train and test images
    TRAIN_DIR = r'D:\UserD\Li\FSCE-1\datasets\my_dataset\image' + "\\"
    TRAIN_CSV_PATH = r'D:\UserD\Li\FSCE-1\datasets\my_dataset_555\annotations\instances_val.json'

    # Glob the directories and get the lists of train and test images
    train_fns = glob.glob(TRAIN_DIR + '*')
    print('数据集图片数量: {}'.format(len(train_fns)))


    '数据整体分布情况'
    # 分析训练集数据
    generate_anno_eda(TRAIN_DIR, TRAIN_CSV_PATH)

    '图像分辨率'
    # 读取训练集标注文件
    with open(TRAIN_CSV_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_fig = pd.DataFrame(train_data['images'])

    train_fig.head()

    ps = np.zeros(len(train_fig))
    for i in range(len(train_fig)):
        # ps[i] = train_fig['width'][i] * train_fig['height'][i] / 1e6
        ps[i] = train_fig['width'][i] * train_fig['height'][i]
    plt.title('训练集图片大小分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)
    plt.show()



    train_anno = pd.DataFrame(train_data['annotations'])
    df_train = pd.merge(left=train_fig, right=train_anno, how='inner', left_on='id', right_on='image_id')
    df_train['bbox_xmin'] = df_train['bbox'].apply(lambda x: x[0])
    df_train['bbox_ymin'] = df_train['bbox'].apply(lambda x: x[1])
    df_train['bbox_w'] = df_train['bbox'].apply(lambda x: x[2])
    df_train['bbox_h'] = df_train['bbox'].apply(lambda x: x[3])
    df_train['bbox_xcenter'] = df_train['bbox'].apply(lambda x: (x[0] + 0.5 * x[2]))
    df_train['bbox_ycenter'] = df_train['bbox'].apply(lambda x: (x[1] + 0.5 * x[3]))


    '图像亮度分析'
    # images_df = pd.DataFrame(df_train.file_name.unique())
    # images_df.columns = ['file_name']
    # brightness_df = add_brightness(images_df)
    # # brightness_info = collections.Counter(brightness_df["brightness"].values.astype(np.uint8))
    # # plt.title('训练集亮度分布', fontproperties=myfont)
    # # sns.distplot(brightness_df["brightness"].values.astype(np.uint8), bins=21, kde=False)
    # # plt.show()
    # # brightness_df.head()
    #
    # dark_names = brightness_df[brightness_df['brightness'] < 50].file_name
    # plot_image_examples(df_train[df_train.file_name.isin(dark_names)], title='暗图片')
    #
    # bright_names = brightness_df[brightness_df['brightness'] > 130].file_name
    # plot_image_examples(df_train[df_train.file_name.isin(bright_names)], title='亮图片')
    # plt.show()
    #
    # sns.set(rc={'figure.figsize': (12, 6)})
    # ps = np.zeros(len(brightness_df))
    # for i in range(len(brightness_df)):
    #     ps[i] = brightness_df['brightness'][i]
    # plt.title('图片亮度分布' ,fontproperties=myfont)
    # sns.distplot(ps, bins=21, kde=False)
    # plt.show()

    '目标分布分析'
    ps = np.zeros(len(df_train))
    for i in range(len(df_train)):
        ps[i] = df_train['area'][i]
        # ps[i] = df_train['area'][i] / 1e6
    plt.title('训练集目标大小分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)
    plt.show()

    # 各类别目标形状分布
    sns.set(rc={'figure.figsize': (12, 6)})
    sns.relplot(x="bbox_w", y="bbox_h", hue="category_id", col="category_id", data=df_train)
    # sns.relplot(x="bbox_w", y="bbox_h", hue="category_id", col="category_id", data=df_train[0:1000])
    plt.show()

    # 各类别目标中心点形状分布
    sns.set(rc={'figure.figsize': (12, 6)})
    sns.relplot(x="bbox_xcenter", y="bbox_ycenter", hue="category_id", col="category_id", data=df_train)
    # sns.relplot(x="bbox_xcenter", y="bbox_ycenter", hue="category_id", col="category_id", data=df_train[0:1000])
    plt.show()

    sns.set(rc={'figure.figsize': (12, 6)})
    plt.title('训练集目标大小分布', fontproperties=myfont)
    sns.violinplot(x=df_train['category_id'], y=df_train['area'])
    plt.show()

    sns.set(rc={'figure.figsize': (12, 6)})
    plt.title('训练集小目标分布', fontproperties=myfont)
    plt.ylim(0, 4000)
    sns.violinplot(x=df_train['category_id'], y=df_train['area'])
    plt.show()

    sns.set(rc={'figure.figsize': (12, 6)})
    plt.title('训练集大目标分布', fontproperties=myfont)
    plt.ylim(10000, max(df_train.area))
    sns.violinplot(x=df_train['category_id'], y=df_train['area'])
    df_train.area.describe()
    plt.show()

    graph = sns.countplot(data=df_train, x='category_id')
    graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
    plt.title('各类别目标数量分布', fontproperties=myfont)
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x() + p.get_width() / 2., height + 0.1, height, ha="center")
    plt.show()

    '重点图片分析'
    # 单张图片目标数量分布
    df_train['bbox_count'] = df_train.apply(lambda row: 1 if any(row.bbox) else 0, axis=1)
    train_images_count = df_train.groupby('file_name').sum().reset_index()
    train_images_count['bbox_count'].describe()
    # 目标数量超过50个的图片
    train_images_count['file_name'][train_images_count['bbox_count']>50]

    # 目标数量超过100个的图片
    train_images_count['file_name'][train_images_count['bbox_count'] > 50]

    less_spikes_ids = train_images_count[train_images_count['bbox_count'] > 50].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='单图目标超过50个（示例）')
    less_spikes_ids = train_images_count[train_images_count['bbox_count'] > 100].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='单图目标超过100个（示例）')
    less_spikes_ids = train_images_count[train_images_count['bbox_count'] < 5].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='单图目标少于5个（示例）')

    # 单图目标覆盖分析
    less_spikes_ids = train_images_count[train_images_count['area'] > max(train_images_count['area']) * 0.9].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标总面积最大（示例）')
    less_spikes_ids = train_images_count[train_images_count['area'] < min(train_images_count['area']) * 1.1].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标总面积最小（示例）')

    # 超大/极小目标分析
    df_train['bbox_count'] = df_train.apply(lambda row: 1 if any(row.bbox) else 0, axis=1)
    train_images_count = df_train.groupby('file_name').max().reset_index()
    less_spikes_ids = train_images_count[train_images_count['area'] > max(train_images_count['area']) * 0.8].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='单目标面积最大（示例）')

    df_train['bbox_count'] = df_train.apply(lambda row: 1 if any(row.bbox) else 0, axis=1)
    train_images_count = df_train.groupby('file_name').min().reset_index()
    less_spikes_ids = train_images_count[train_images_count['area'] > min(train_images_count['area']) * 1.2].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='单目标面积最小（示例）')

    '目标遮挡分析'
    file_list = df_train['file_name'].unique()
    train_iou_cal = pd.DataFrame(columns=('file_name', 'iou_agg', 'iou_cnt'))
    for i in range(len(file_list)):
        tmp = df_train['bbox'][df_train.file_name == file_list[i]].reset_index(drop=True)
        iou_agg, iou_cnt = bbox_iou(tmp)
        train_iou_cal.loc[len(train_iou_cal)] = [file_list[i], iou_agg, iou_cnt]

    train_iou_cal.iou_agg.describe()

    ps = np.zeros(len(train_iou_cal))
    for i in range(len(train_iou_cal)):
        ps[i] = train_iou_cal['iou_agg'][i]
    plt.title('训练集目标遮挡程度分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)

    train_iou_cal.iou_cnt.describe()

    ps = np.zeros(len(train_iou_cal))
    for i in range(len(train_iou_cal)):
        ps[i] = train_iou_cal['iou_cnt'][i]
    plt.title('训练集目标遮挡数量分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)

    less_spikes_ids = train_iou_cal[train_iou_cal['iou_agg'] > max(train_iou_cal['iou_agg']) * 0.9].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标遮挡程度最高（示例）')
    less_spikes_ids = train_iou_cal[train_iou_cal['iou_agg'] <= min(train_iou_cal['iou_agg']) * 1.1].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标遮挡程度最低（示例）')
    less_spikes_ids = train_iou_cal[train_iou_cal['iou_cnt'] > max(train_iou_cal['iou_cnt']) * 0.9].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标遮挡数量最高（示例）')
    less_spikes_ids = train_iou_cal[train_iou_cal['iou_cnt'] <= min(train_iou_cal['iou_cnt']) * 1.1].file_name
    plot_image_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标遮挡数量最低（示例）')

    '颜色分析'
    # 图像RGB分布
    files = os.listdir(TRAIN_DIR)

    R = 0.
    G = 0.
    B = 0.
    R_2 = 0.
    G_2 = 0.
    B_2 = 0.
    N = 0

    for f in files:
        img = cv2.imread(TRAIN_DIR + f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        h, w, c = img.shape
        N += h * w

        R_t = img[:, :, 0]
        R += np.sum(R_t)
        R_2 += np.sum(np.power(R_t, 2.0))

        G_t = img[:, :, 1]
        G += np.sum(G_t)
        G_2 += np.sum(np.power(G_t, 2.0))

        B_t = img[:, :, 2]
        B += np.sum(B_t)
        B_2 += np.sum(np.power(B_t, 2.0))

    R_mean = R / N
    G_mean = G / N
    B_mean = B / N

    R_std = np.sqrt(R_2 / N - R_mean * R_mean)
    G_std = np.sqrt(G_2 / N - G_mean * G_mean)
    B_std = np.sqrt(B_2 / N - B_mean * B_mean)

    print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))
    print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))
    # 目标RGB分析
    # 可能遇到jupyter输出内存报错
    from tqdm import tqdm

    df_train['r_channel'] = 0
    df_train['g_channel'] = 0
    df_train['b_channel'] = 0
    for i in tqdm(df_train.index):
        array = bb_rgb_cal(Image.open(TRAIN_DIR + str(df_train.file_name[i])), df_train.bbox[i])
        df_train['r_channel'].at[i] = array[0]
        df_train['g_channel'].at[i] = array[1]
        df_train['b_channel'].at[i] = array[2]

    ps = np.zeros(len(df_train[:10000]))
    for i in range(len(df_train[:10000])):
        ps[i] = df_train['r_channel'][df_train.category_id == 1][i]
    plt.title('类别1目标r_channel分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)

    ps = np.zeros(len(df_train[:10000]))
    for i in range(len(df_train[:10000])):
        ps[i] = df_train['g_channel'][df_train.g_channel > 0][df_train.category_id == 1][i]
    plt.title('类别1目标g_channel分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)

    ps = np.zeros(len(df_train[:10000]))
    for i in range(len(df_train[:10000])):
        ps[i] = df_train['b_channel'][df_train.b_channel > 0][df_train.category_id == 1][i]
    plt.title('类别1目标b_channel分布', fontproperties=myfont)
    sns.distplot(ps, bins=21, kde=False)

    # 灰度图效果
    less_spikes_ids = train_iou_cal[train_iou_cal['iou_cnt'] > max(train_iou_cal['iou_cnt']) * 0.8].file_name
    plot_gray_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标遮挡数量最高（灰度）')
    less_spikes_ids = train_iou_cal[train_iou_cal['iou_cnt'] <= min(train_iou_cal['iou_cnt']) * 1.1].file_name
    plot_gray_examples(df_train[df_train.file_name.isin(less_spikes_ids)], title='目标遮挡数量最低（灰度）')






