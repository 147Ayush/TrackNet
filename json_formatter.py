def format_to_json(detections):
    result = []
    for obj in detections:
        obj_json = {
            "object": obj["name"],
            "id": obj["id"],
            "bbox": obj["bbox"],
            "subobject": []
        }


        for sub_obj in obj["sub_objects"]:
            obj_json["subobject"].append({
                "object": sub_obj["name"],
                "id": sub_obj["id"],
                "bbox": sub_obj["bbox"]
            })

        result.append(obj_json)
    return result
