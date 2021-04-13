#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#

# Defined Constants
import os
sep = os.path.sep
cwd = os.getcwd()

# models/fault/model/model.py

models_path = cwd + sep + 'models'
fseg_xml = models_path + sep + 'fault' + sep + 'model' + sep + 'shp-fseg-60.xml'
fseg_bin = models_path + sep + 'fault' + sep + 'model' + sep +  'shp-fseg-60.bin'

salt_xml = models_path + sep + 'salt' + sep + 'model' + sep +  'saved_model.xml'
salt_bin = models_path + sep + 'salt' + sep + 'model' + sep +  'saved_model.bin'

facies_xml = models_path + sep + 'facies' + sep + 'model' + sep +  'shp-deco-skip.xml'
facies_bin = models_path + sep + 'facies' + sep + 'model' + sep +  'shp-deco-skip.bin'

paths_dict = {
    'fseg': (fseg_xml, fseg_bin),
    'salt': (salt_xml, salt_bin),
    'facies': (facies_xml, facies_bin)
}

demo_paths_for_imports = {
    'fseg': 'models.fault.model',
    'salt': 'models.salt.model',
    'facies': 'models.facies.model'
}
