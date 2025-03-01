

class ObjectPrompt:
    def __init__(self, obj_id:int):
        self.object_id = obj_id
        self.input_position = []
        self.input_label = []
        self.frame_ids = []

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
        self.frame_id = -1

    def getId(self):
        return self.object_id
    
    def addFrameId(self, frame_id):
        self.frame_ids.append(frame_id)
    
    def hasFrameId(self, frame_id):
        if (frame_id in self.frame_ids):
            return True
        
        return False