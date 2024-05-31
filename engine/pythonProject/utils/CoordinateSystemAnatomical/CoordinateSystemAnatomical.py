from utils.CoordinateSystem.CoordinateSystem import CoordinateSystem

class CoordinateSystemAnatomical(CoordinateSystem):
    """
    Add the anatomical axes as properties.
    """
    def __init__(self):
        super().__init__()
        self.ML = []
        self.PA = []
        self.IS = []
    
    def set_PA(self, value):
        self.set_xAxis(value)
        self.PA = value
    
    def set_IS(self, value):
        self.set_yAxis(value)
        self.IS = value
        
    def set_ML(self, value):
        self.set_zAxis(value)
        self.ML = value
        