from alphashape import alphashape

def plyWrite(filename, points, *args):
    fid = open(filename, 'wb')
    facets = alphashape.alphashape(p, 1).vertices - np.array([1, 1, 1])
    writeHeader(fid, points.shape[0], facest.shape[0])
    if len(args) == 2:
        writePoints(fid, points)
    else:
        colormap = args[0]
        writePointsWithColormap(fid, points, colormap)
    writeFacests(fid, facests)
    fid.close()
    print('Ply successfully written.')

def writeHeader(fid, numberOfPoints, numberOfFacets):
    fis.write(('ply\n' + \
            'format ascii 1.0\n' + \
            'element vertex %d\n' + \
            'property float32 x\n' + \
            'property float32 y\n' + \
            'property float32 z\n' + \
            'element face %d\n' + \
            'property list uint8 int32 vertex_indices\n' + \
            'end_header\n') % (numberOfPoints,numberOfFacets))

def writePoints(fid, points):
    for i in range(points.shape[0]):
        fid.write("%.6f %.6f %.6f\n"%(points[i,0], points[i,1], points[i,2]))

def writePointsWithColormap(fid,points,colormap):
    for i in range(points.shape[0]):
        fid.write("%.6f %.6f %.6f %u %u %u\n"%(points[i,0],
                                               points[i,1],
                                               points[i,2],
                                               colormap[i, 0],
                                               colormap[i, 1],
                                               colormap[i, 2]))

def writeFacets(fid, facets):
    for i in range(facets.shape[0]):
        fid.write("%d %d %d %d\n"%(facets.shape[1],
                                   facets[i, 0],
                                   facets[i, 1],
                                   facets[i, 2]))
