###############################################################################
# Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
###############################################################################

#!/bin/bash

import numpy as np
import pandas as pd
import json

def imagenet_labels():
    ''' Return a list of imagenet_labels '''
    imagenet_labels = None
    imagenet_index = open('./imagenet_class_index.json')
    if imagenet_index:
        imagenet_labels = json.load(imagenet_index)
        imagenet_index.close()

    return imagenet_labels

class QAicSoftmax:
    def __init__(self, path, needSoftmax):
        self.softmax = np.fromfile(path, np.float32)
        if (needSoftmax == True):
            self.softmax = np.exp(self.softmax) / np.sum(np.exp(self.softmax), axis=0)

    def shape(self):
        return self.softmax.shape

    def topk(self, k):
        '''Get topk results sorted by highest confidence'''
        topk = {}

        for idx, score in enumerate(self.softmax):
           topk[idx] = score

        if len(topk) == 0:
           return None

        # Sort highest confidence first
        topk = sorted(topk.items(), key=lambda x: x[1], reverse=True)

        return topk[:k] if k else topk[:5]


def pdviewer(binpath, k, needSoftmax=False):
    qaic_softmax = QAicSoftmax(binpath, needSoftmax)
    print('Softmax dimensions: {}'.format(qaic_softmax.shape()))
    topk = qaic_softmax.topk(k)

    labels = imagenet_labels()

    df = pd.DataFrame()

    results = {'Top-K': [], 'Index': [], 'Class': [], 'Confidence': []}

    k_idx = 1
    for idx, confidence in topk:
       if idx==1000:
          continue
       label = labels[str(idx)][1]
       results['Top-K'].append('K{}'.format(k_idx))
       results['Index'].append(idx)
       results['Class'].append(label)
       results['Confidence'].append(confidence)
       k_idx += 1
        
    print('Top ' + str(k) + ' matches:')
    return pd.DataFrame(results)

df = pdviewer('./resnetv17_dense0_fwd-activation-0-inf-1.bin', 5, True)
print(df, end='\n\n')

