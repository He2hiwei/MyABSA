
import numpy as np

DEFAULT_FILE_PATH = "/Users/apple/AVEC2017/utils/datasets/glove.840B.300d.txt"

def loadWordVectors(tokens, filepath=DEFAULT_FILE_PATH, dimensions=300):
    """Read pretrained GloVe vectors"""
    
    wordVectors = np.random.uniform(0.25,-0.25,(len(tokens), dimensions))
    with open(filepath) as ifs:
        for line in ifs:
            line = line.strip()
            if not line:
                continue
            row = line.split(' ')
            token = row[0]
            if token not in tokens:
                continue
            
            data = [float(x) for x in row[1:]]
            
#            try:
#                data = [float(x) for x in row[1:]]
#            except ValueError:
#                print('token: %s, ValueError' % token)
##                data = [float('0.0') if x == '.' else float(x) for x in row[1:]]
#                data = np.random.uniform(0.25,-0.25,(1, dimensions))
     
            if len(data) != dimensions:
                print('token: %s, wrong number of dimensions' % token)
                data = np.random.uniform(0.25,-0.25,(1, dimensions))
#                raise RuntimeError("wrong number of dimensions")
            wordVectors[tokens[token]] = np.asarray(data)
    return wordVectors
