import xml.etree.cElementTree as ET

def CreateNewASAP_Annotations(annotation_list, group_name_list=['detected_metastases'], annotation_color='#00FF00', group_color='#FF0000'):
    '''
    Creates and returns ET.ElementTree Object that can be used to write to file
    Takes in a list of 'coordinate lists' and adds each list as <Annotation>
    '''
    asap_annotation = ET.Element('ASAP_Annotations')
    annotations = ET.SubElement(asap_annotation, 'Annotations')
    annotation_groups = ET.SubElement(asap_annotation, 'AnnotationGroups')

    AddAnnotationElements(annotations, annotation_list, color=annotation_color)
    AddGroupElements(annotation_groups, group_name_list, group_color)

    return ET.ElementTree(asap_annotation)

def AppendASAP_AnnotationsFile(filename, annotation_list, group_name_list=['detected_metastases'], annotation_color='#00FF00', group_color='#FF0000'):
    '''
    Adds given coords and group name to xml file and returns ElementTree object that can be used to write to file
    '''
    tree = ET.parse(filename)
    annotations = tree.find('Annotations')
    annotation_groups = tree.find('AnnotationGroups')

    AddAnnotationElements(annotations, annotation_list, color=annotation_color)
    AddGroupElements(annotation_groups, group_name_list, group_color)

    return tree


def AddAnnotationElements(parent, annotation_list, group='detected_metastases', color='#00FF00'):
    '''
    Adds 'Annotation' sub tree under given parent element of ElementTree Class

    parent: parent element to add all given annotations under
    annotation_list: list of annotations such that each annotation consists of list of coordinates
                        (essentially a list of lists is expected here)
    group: group attribute to make annotations part of
    color: color
    '''
    for i in range(len(annotation_list)):
        annotation_element = ET.SubElement(parent, 'Annotation')
        annotation_element.set('Color', color)
        annotation_element.set('Name', 'Annotation '+str(i))
        annotation_element.set('PartOfGroup', group)
        annotation_element.set('Type', 'Polygon')

        coordinates = ET.SubElement(annotation_element, 'Coordinates')

        AddCoordinateElement(coordinates, annotation_list[i])


def AddCoordinateElement(coordinates_parent_element, coordinate_list):
    for c in range(len(coordinate_list)):
        coordinate = ET.SubElement(coordinates_parent_element, 'Coordinate')
        coordinate.set('Order', str(c))
        coordinate.set('Y', str(coordinate_list[c][0]))
        coordinate.set('X', str(coordinate_list[c][1]))

def AddGroupElements(parent, group_name_list, group_color='#FF00FF'):
    for group_name in group_name_list:
        group = ET.SubElement(parent, 'Group')
        group.set('Name', group_name)
        group.set('Color', group_color)
        group.set('PartOfGroup', 'None')
        attr = ET.SubElement(group, 'Attributes')
