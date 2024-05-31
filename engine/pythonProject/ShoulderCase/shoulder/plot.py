def plot(obj, colors, *args):
    if "wireframe" in args:
        obj.scapula.plotLandmarksWireframe(colors[0], colors[1])
    if "scapulaSurface" in args:
        obj.scapula.plotSurface(colors[0], colors[1])
    if "coordinateSystem" in args:
        obj.scapula.coordSys.plot(colors[1])
    if "centeredCoordinateSystem" in args:
        obj.scapula.coordSys.plot(colors[1])
        
        
        
        
        