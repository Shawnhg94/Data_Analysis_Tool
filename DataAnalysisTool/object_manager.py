import yaml
from object_entity import ObjectEntity

class ObjectManager:
    def __init__(self):
        with open("DataAnalysisTool/object.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        self.obj_entity_map = {}
        for ele in config['DRIVING_objects']:
            name = ele['Entity']
            id = ele['ID']
            colour = ele['Colour']
            entity = ObjectEntity(name, id=id, colour=colour)
            self.obj_entity_map.update({name: entity})
        
        # Oject Label Maps
        self.object_labels = {}

    def set(self, obj_index: int, obj_name:str):
        entity = self.obj_entity_map[obj_name]
        self.object_labels.update({obj_index: entity})
        print("set object label: {}", self.object_labels)

    def unset(self, obj_index: int):
        entity = self.object_labels.pop(obj_index)
        print("unset object label: {}", entity)

    def get_object_lists(self):
        # print(list(self.obj_entity_map.keys()))
        return list(self.obj_entity_map.keys())
    
    def get_entity_id(self, obj_index: int):
        return self.object_labels[obj_index].id
    
    
    def get_entity_colour(self, obj_index: int):
        if (obj_index in self.object_labels):
            return self.object_labels[obj_index].colour
        return None