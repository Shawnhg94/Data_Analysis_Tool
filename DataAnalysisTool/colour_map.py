
object_colour_map = [(110, 9, 150), (50, 168, 82), (35, 178, 196), (176, 184, 37), (4, 68, 133), (17, 22, 120), (60, 24, 181),  (133, 45, 122), (6, 108, 156), (97, 16, 179)]

def colour_map(object_id: int):
    colour_len = len(object_colour_map)

    if object_id >= colour_len:
        return object_colour_map[0]
    
    return object_colour_map[object_id]
