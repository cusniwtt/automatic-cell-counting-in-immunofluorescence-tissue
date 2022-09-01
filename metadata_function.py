import os
import fnmatch

def getPath(path, type='DAPI'):
    paths = []
    regex = '*' + type + '.tif'
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, regex):
            paths.append(file)
    return paths