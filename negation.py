class Cues:
    def __init__(self, data):
        self.sentences = data[0]
        self.cues = data[1]
        self.num_sentences = len(data[0])
class Scopes:
    def __init__(self, data):
        self.sentences = data[0]
        self.cues = data[1]
        self.scopes = data[2]
        self.num_sentences = len(data[0])
