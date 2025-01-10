def map_subobjects(objects, relationships):
    for obj in objects:
        for rel in relationships:
            if obj["name"] == rel["parent"]:
                for sub_obj in objects:
                    if sub_obj["name"] == rel["child"] and is_within(sub_obj["bbox"], obj["bbox"]):
                        obj["sub_objects"].append(sub_obj)
    return objects

def is_within(child_bbox, parent_bbox):
    x1_c, y1_c, x2_c, y2_c = child_bbox
    x1_p, y1_p, x2_p, y2_p = parent_bbox
    return x1_c >= x1_p and y1_c >= y1_p and x2_c <= x2_p and y2_c <= y2_p
