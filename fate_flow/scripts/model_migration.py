import argparse
import base64
import json
import os

from arch.api import session
from arch.api.model_manager.manager import get_model_table_partition_count
from arch.api.utils.core import json_loads, json_dumps

from federatedml.protobuf.generated import pipeline_pb2


path = './'
WORK_MODE = 1


# used by import
IMPORT_ROLE = {
    "guest": [9999],
    "host": [10000],
    "arbiter": [10000]
}


def export_model(name, namespace):
    session.init(mode=WORK_MODE)
    model_data = {}
    meta_data = {}
    try:
        pipeline_model_table = session.table(name=name, namespace=namespace,
                                             partition=get_model_table_partition_count(),
                                             create_if_missing=False, error_if_exist=False)

        meta_data = pipeline_model_table.get_metas()
        for storage_key, buffer_object_bytes in pipeline_model_table.collect(use_serialize=False):
            model_data[storage_key] = base64.b64encode(buffer_object_bytes).decode()
    except:
        print('get model data failed')
    session.stop()
    return model_data, meta_data


def import_model(name, namespace, model_data, meta_data):
    session.init(mode=WORK_MODE)
    try:
        pipeline_model_table = session.table(name=name, namespace=namespace,
                                             partition=get_model_table_partition_count(),
                                             create_if_missing=False, error_if_exist=False)
        for storage_key, buffer_object in model_data.items():
            buffer_object_bytes = base64.b64decode(buffer_object.encode())
            if storage_key == 'pipeline.pipeline:Pipeline':
                pipeline = pipeline_pb2.Pipeline()
                pipeline.ParseFromString(buffer_object_bytes)
                train_runtime_conf = json_loads(pipeline.train_runtime_conf)
                train_runtime_conf['role'] = IMPORT_ROLE
                pipeline.train_runtime_conf = json_dumps(train_runtime_conf, byte=True)
                buffer_object_bytes = pipeline.SerializeToString()
            pipeline_model_table.put(storage_key, buffer_object_bytes, use_serialize=False)
        session.save_data_table_meta(meta_data, data_table_namespace=namespace, data_table_name=name)
    except:
        print('import model data failed')
    session.stop()


def model_handle(way, name, namespace, role, party_id):
    model_path = os.path.join(path, 'model_data', role, namespace, name, 'pipeline')
    if way == 'export':
        model_data, meta_data = export_model(name, gen_party_model_id(namespace, role, party_id))
        dump_model(model_data, model_path)
        dump_model(meta_data, model_path.replace('pipeline', 'meta'))
        print('export model success, model path is "{}"'.format(model_path))
    elif way == 'import':
        model_data = get_model(model_path)
        meta_data = get_model(model_path.replace('pipeline', 'meta'))
        import_model(name, gen_party_model_id(namespace, role, party_id), model_data, meta_data)
        print('import model success')
    else:
        raise Exception('Only supports import and export operations')


def dump_model(config_data, model_path):
    if not config_data:
        raise Exception('model is null')
    if os.path.exists(model_path):
        raise Exception('the model already exists, path is "{}"'.format(model_path))
    try:
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        with open(model_path, "w") as f:
            json.dump(config_data, f, indent=4)
    except:
        raise EnvironmentError("dump model file from '{}' failed!".format(model_path))


def get_model(model_path):
    if not os.path.exists(model_path):
        raise Exception('no find this model file:"{}"'.format(model_path))
    try:
        with open(model_path) as f:
            return json.load(f)
    except:
        raise EnvironmentError("loading model file from '{}' failed!".format(model_path))


def gen_party_model_id(model_id, role, party_id):
    return '#'.join([role, str(party_id), model_id])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--way', required=True, type=str, help="export or import")
    parser.add_argument('-i', '--model_id', required=True, type=str, help="model id")
    parser.add_argument('-v', '--model_version', required=True, type=str, help="model version")
    parser.add_argument('-r', '--role', required=True, type=str, help="role")
    parser.add_argument('-p', '--party_id', required=True, type=int, help="party id")
    args = parser.parse_args()
    handle_way = args.way
    namespace = args.model_id
    name = args.model_version
    role = args.role
    party_id = args.party_id
    model_handle(handle_way, name, namespace, role, party_id)







