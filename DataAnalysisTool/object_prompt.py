

class ObjectPrompt:
    def __init__(self, obj_id:int):
        self.object_id = obj_id
        self.input_position = []
        self.input_label = []

    def addPrompt(self, position: tuple, label:int):
        self.input_position.append(position)
        self.input_label.append(label)

    def isActivate(self):
        if len(self.input_position) > 0:
            return True
        return False
    
    def clear(self):
        self.input_label.clear()
        self.input_position.clear()

    def getId(self):
        return self.object_id
    