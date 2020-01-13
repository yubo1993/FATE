#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging.config
import os

import yaml

LOGGER = None


def set_logger(config_path):
    global LOGGER
    with open('../conf/logging.yaml', 'r') as f:
        yaml_cfg = yaml.safe_load(f.read())
    logging.config.dictConfig(yaml_cfg)
    try:
        LOGGER = logging.getLogger("submit")
    except KeyError as e:
        LOGGER = logging.getLogger()


def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


maybe_mkdir("../logs")
set_logger("../conf/logging.yaml")
