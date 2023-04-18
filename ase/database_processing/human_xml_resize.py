import xml.etree.ElementTree as ET

body_names_height = ["pelvis", "torso", "head", "headset",
                     "right_upper_arm",
                     "left_upper_arm",
                     "right_shin", "right_foot",
                     "left_shin", "left_foot"]

body_names_arm = ["right_lower_arm", "right_hand",
                  "left_lower_arm", "left_hand"]

body_names_extra_height = ["torso"]
body_names_minus_height = ["headset"]


geom_names_height = ["pelvis", "upper_waist", "torso",
                     "right_clavicle", "left_clavicle",
                     "head",
                     "right_thigh", "right_shin", "right_foot",
                     "left_thigh", "left_shin", "left_foot"]

site_names_height = ["pelvis", "upper_waist", "torso", "head",
                     "right_foot",
                     "left_foot"]

site_names_geom = ["right_thigh", "left_thigh"]
site_names_geom_half = ["right_shin", "left_shin",]


def str_to_float_list(str):
    return [float(v) for v in str.split(" ")]


def float_list_to_val_str(float_list):
    # first delete decimals when not needed
    val_list = [int(v) if v == 0.0 else v for v in float_list]
    str_list = [str(v) for v in val_list]
    return " ".join(str_list)


def modify_pos(tree, factor, axis):
    pos = str_to_float_list(tree.get("pos"))
    new_val = pos[axis] * factor
    new_val = round(new_val, 3)
    pos[axis] = new_val
    new_pos = float_list_to_val_str(pos)
    tree.set("pos", new_pos)


def add_displ_to_pos(tree, displacement, axis):
    pos = str_to_float_list(tree.get("pos"))
    new_val = pos[axis] + displacement
    new_val = round(new_val, 3)
    pos[axis] = new_val
    new_pos = float_list_to_val_str(pos)
    tree.set("pos", new_pos)


def modify_site_from_geom(site_tree, modified_geom_tree, axis, length_prc):
    fromto = str_to_float_list(modified_geom_tree.get("fromto"))
    pos = str_to_float_list(site_tree.get("pos"))
    size = str_to_float_list(site_tree.get("size"))
    #modifies pos
    midpoint = (fromto[axis+3] + fromto[axis]) / 2
    pos[axis] = midpoint
    new_pos = float_list_to_val_str(pos)
    site_tree.set("pos", new_pos)
    #modifies size
    length = abs(fromto[axis+3] - fromto[axis])
    size[axis-1] = length * length_prc + 0.001
    new_size = float_list_to_val_str(size)
    site_tree.set("size", new_size)


def modify_from_to(tree, factor, axis):
    fromto = str_to_float_list(tree.get("fromto"))
    new_val_from = fromto[axis] * factor
    new_val_from = round(new_val_from, 3)
    new_val_to = fromto[axis+3] * factor
    new_val_to = round(new_val_to, 3)
    fromto[axis] = new_val_from
    fromto[axis+3] = new_val_to
    new_fromto = float_list_to_val_str(fromto)
    tree.set("fromto", new_fromto)


def write_xml(xmlTree, filename):
    xmlTree.write(filename)


def modify_els(worldbody, factor_height, factor_arm, height_displ):
    for child in worldbody.iter('body'):
        if child.attrib["name"] in body_names_height:
            modify_pos(child, factor_height, 2)
        elif child.attrib["name"] in body_names_arm:
            modify_pos(child, factor_arm, 2)

        if child.attrib["name"] in body_names_extra_height:
            add_displ_to_pos(child, height_displ, 2)
        elif child.attrib["name"] in body_names_minus_height:
            add_displ_to_pos(child, -height_displ, 2)

    for child in worldbody.iter('geom'):
        if child.attrib["name"] in geom_names_height:
            if "fromto" in child.attrib.keys():
                modify_from_to(child, factor_height, 2)
            else:
                modify_pos(child, factor_height, 2)
    # Important to make sites after geom because some of them need modified values
    for child in worldbody.iter('site'):
        if child.attrib["name"] in site_names_height:
            modify_pos(child, factor_height, 2)
        elif child.attrib["name"] in site_names_geom or child.attrib["name"] in site_names_geom_half:
            respective_geom = None
            for geom_child in worldbody.iter('geom'):
                if geom_child.attrib["name"] == child.attrib["name"]:
                    respective_geom = geom_child
            if child.attrib["name"] in site_names_geom:
                modify_site_from_geom(child, respective_geom, 2, 1.0)
            else:
                modify_site_from_geom(child, respective_geom, 2, 0.5)

    for child in worldbody.iter('camera'):
        if child.attrib["name"] == "egocentric":
            modify_pos(child, factor_height, 2)

def main():
    ref_height = 1.4975
    ref_arm_length = 0.534
    ref_extra_height = 0.0084
    arm_to_height_ratio = 0.54 / 1.8
    desired_heights = [1.4, 1.52, 1.6, 1.68, 1.8, 1.85, 1.93, 2.07, 2.12, 2.2]
    for height in desired_heights:
        arm_length = arm_to_height_ratio * height
        height_factor = height / ref_height
        arm_factor = arm_length / ref_arm_length
        extra_height = ref_extra_height * height_factor
        # print("h", height, "a", arm_length, "hf", height_factor, "af", arm_factor, "eh", extra_height)
        tree = ET.parse('ase/data/assets/mjcf/amp_humanoid_vrh_ref.xml')
        root = tree.getroot()
        worldbody = root[3]
        modify_els(worldbody, height_factor, arm_factor, extra_height)
        height_name = '{:.2f}'.format(height).replace(".", "")
        filename = f'ase/data/assets/mjcf/amp_humanoid_vrh_{height_name}.xml'
        write_xml(tree, filename)


if __name__ == "__main__":
    main()