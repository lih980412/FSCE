from xml.dom.minidom import Document
import os
import cv2



def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset

# def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    """
    dic = {
        '边部裂口': 'edgeBreach',
        '边部裂纹': 'edgeCrack',
        '边部翘皮': 'edgeUpwarping',
        '边线': 'edgeLine',
        '擦伤': 'scratch',
        '擦伤铁皮': 'scratchIronSheet',
        '除磷铁皮': 'dephosphorizationIronSheet',
        '大翘皮': 'bigUpwarping',
        '点状缺陷': 'spotty',
        '辊系铁皮': 'rollerIronSheet',
        '辊印': 'rollMark',
        '红铁皮': 'redIronSheet',
        '划伤': 'strike',
        '机清条纹': 'stripe',
        '夹渣': 'slagInclusion',
        '夹渣轻': 'slagInclusion',
        '孔洞': 'hole',
        '垃圾和粘铁': 'garbage',
        '面翘皮': 'surfaceUpwarping',
        '热裂纹': 'heatingCrack',
        '水': 'water',
        '铁皮灰轻': 'tinyIronSheetDust',
        '铁皮灰重': 'denseIronSheetDust',
        '铜脆': 'copperFragile',
        '伪缺陷': 'pseudoDefect',
        '温度铁皮': 'temperatureIronSheet',
        '轧破': 'rollBroken',
        '氧化铁皮': 'oxidationIronSheet',
        '异物压入': 'abnormalInclusion',
        '折痕': 'fold'
    }
    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        if name.endswith("jpg") or name.endswith("xml") or name.endswith("py"):
            continue
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name, encoding="utf-8", errors="ignore")
        txtList = txtFile.readlines()
        img = cv2.imread(picPath + name[0:-4] + ".jpg")
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("边部翘皮")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic["边部翘皮"])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')

        f.close()


if __name__ == "__main__":
    picPath = r"K:\Li\123/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = r"K:\Li\123/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = r"K:\Li\123/"  # xml文件保存路径，后面的/一定要带上
    makexml(picPath, txtPath, xmlPath)