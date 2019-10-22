import xml.etree.ElementTree as ET

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def xml2dict(xml_name, class_list):
    objects = ET.parse(xml_name).findall("object")

    dicts = {'__background__':[]}
    for cls_name in class_list:
        if cls_name != '__background__':
            dicts = Merge(dicts, dict([(cls_name, [])]))

    for object in objects:
        class_name = object.find('name').text.strip()
        class_name = class_name.upper()
        xmin = int(object.find('bndbox/xmin').text)
        ymin = int(object.find('bndbox/ymin').text)
        xmax = int(object.find('bndbox/xmax').text)
        ymax = int(object.find('bndbox/ymax').text)

        if '__background__' == class_name:
            continue
        obj = [xmin, ymin, xmax, ymax, 1]
        dicts[class_name].append(obj)

    return dicts

def det_results2dict(results, class_list):

    dicts = {'__background__':[]}
    for cls_name in class_list:
        if cls_name != '__background__':
            dicts = Merge(dicts, dict([(cls_name, [])]))

    for res in results:
        cls_name = class_list[res[-1]]
        if cls_name != '__background__':
            dicts[] = res[0:5]


if __name__ == "__main__":
    xml_name = "/data/zhangcc/data/SL_around/new/351-H-20190121094220377263_4M_NDJ-20181211QMZ-201904241154152821-black-0-0_new.xml"
    class_list = ('__background__',
                  'TT', 'QZ', 'PP', 'SLK', 'SL', 'SLD', 'SLL', 'M', 'AJ', 'HH', 'SLQ', 'FC', 'GG', 'HM')
    xml2dict(xml_name, class_list)
