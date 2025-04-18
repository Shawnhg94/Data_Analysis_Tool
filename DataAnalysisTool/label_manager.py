import yaml
from object_entity import ObjectEntity

class LabelManager:
    def __init__(self):
        with open("DataAnalysisTool/object.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        self.obj_entity_map = {}
        for ele in config['DRIVING_objects']:
            name = ele['Entity']
            id = ele['ID']
            colour = ele['Colour']
            entity = ObjectEntity(name, id=id, colour=colour)
            self.obj_entity_map.update({id: entity})
    
    def get_colour(self, id:int):
        return self.obj_entity_map[id].colour