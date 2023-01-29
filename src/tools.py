import sys, os, shutil, pickle
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# helper functions for moving files:
def ensureFolder(relativePath):
    if not os.path.exists(relativePath):
        os.makedirs(relativePath)
    return relativePath

def moveFolder(source, destination):
    shutil.move(source, destination)

def copyFile(source, destination):
    shutil.copyfile(source, destination)

def overwritePath(source, target):
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.move(source, target)

def clearFolder(path):
    shutil.rmtree(path)
    ensureFolder(path)

def ensureClear(path):
    if fileExists(path):
        clearFolder(path)
    else:
        ensureFolder(path)
    return path

def fileExists(path):
    return os.path.exists(path)

def removeFile(path):
    os.remove(path)

def removeFolder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def doPickle(arr, path, protocol=4):
    pickle.dump(arr, open(path, 'wb'), protocol=protocol)

def unpickle(path):
    if sys.version_info[0] == 3:
        results = pickle.load(open(path, "rb"), encoding='latin1')
    else:
        results = pickle.load(open(path, "rb"))
    return results


def csvFile(path, delimiter=',', remNewline=False, remQuotes=False):
    file = open(path, 'r')
    lines = file.readlines()
    contents = []
    toStrip = ""
    if remNewline:
        toStrip += "\n"
    if remQuotes:
        toStrip += "\""
    for l in lines:
        comps = l.split(delimiter)
        if len(toStrip):
            for cn in range(len(comps)):
                comps[cn] = comps[cn].strip(toStrip)
        contents.append(comps)
    file.close()
    return contents


def saveAsCsv(array, path, delimiter=','):
    f = open(path, 'w')
    for line in array:
        for vi, value in enumerate(line):
            if vi > 0:
                f.write(delimiter)
            f.write(str(value))
        f.write('\n')
    f.close()
    print('Saved CSV: {}'.format(path))


def saveLatexTable(array, path, roundTo=3):
    f = open(path, 'w')
    for line in array:
        for vi, value in enumerate(line):
            if vi > 0:
                f.write(' & ')
            if isinstance(value, str):
                f.write(value)
            else:
                if float(value) == int(value):
                    f.write(str(int(value)))
                else:
                    f.write(str(round(float(value), ndigits=roundTo)))
        f.write('\\\\\n')
    f.close()
    print('Saved Latex table: {}'.format(path))


def stack(dm, ref=None):
    if type(dm) is np.ndarray:
        return dm
    if ref is not None:
        cat = []
        for d in ref:
            if d in dm:
                cat.append(dm[d])
        return np.array(np.concatenate(cat))
    return np.array(np.concatenate([v for v in list(dm.values()) if len(v)], axis=0))

def unstack(arr, d):
    du = {}
    loc = 0
    for k in d:
        du[k] = arr[loc:loc + len(d[k])]
        loc += len(d[k])
    return du


def filterDataMap(dataMap, prediction, subset):
    fd = {}
    for d in dataMap:
        mask = np.isin(prediction[d], subset)
        if np.sum(mask):
            fd[d] = dataMap[d][mask]
    return fd


def applyMap(dm, labels):
    subset = {}
    for d in dm:
        subset[d] = labels[d][dm[d]]
    return subset


def subsampleDataMap(dataMap, n):
    sum = np.sum([len(dataMap[data]) for data in dataMap])
    prop = float(n) / sum
    dm = {}
    for data in dataMap:
        inds = dataMap[data]
        ns = int(np.ceil(len(inds) * prop))
        inds_sub = np.random.permutation(len(inds))[:ns]
        dm[data] = inds[inds_sub]
    return dm


def expandPrediction(values:dict, dm:dict, nullValue):
    filled = {d: np.repeat(nullValue, len(d.X)) for d in dm}
    for d in values:
        filled[d][dm[d]] = values[d]
    return filled


def repeatMasked(mask, value, nullValue):
    vals = np.array(np.repeat(nullValue, len(mask)), dtype=object)
    vals[mask] = value
    return vals


def commonElements(lists):
    cmn = set(lists[0])
    for l in lists[1:]:
        cmn = cmn.intersection(set(l))
    return list(cmn)


def softmax(X):
    exp_max = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return exp_max / np.sum(exp_max, axis=-1, keepdims=True)