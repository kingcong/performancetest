import os
import time
from bayes_opt import BayesianOptimization

import mindspore
from mindspore import dataset as ds
import datetime

class AutoTune:
    op_type_keys = []
    process_data = {}

    def __init__(self, serialized_data):
        self.process_data = serialized_data

    def internal_autotune_optimize(self, kwargs, is_best_params):
        # print(kwargs)
        data_nums = 1
        origin_data = self.process_data
        temp_data = origin_data['children'][0]
        op_type_worker = origin_data['op_type'] + "1"
        op_type_queue = op_type_worker + "_Queue"
        origin_data['num_parallel_workers'] = int(kwargs[op_type_worker])
        origin_data['connector_queue_size'] = int(kwargs[op_type_queue])

        count = int(len(kwargs)/2)
        for i in range(count-1):
            current_op_type_worker = temp_data['op_type'] + str(i+2)
            current_op_type_queue = current_op_type_worker + str("_Queue")
            temp_data['num_parallel_workers'] = int(kwargs[current_op_type_worker])
            temp_data['connector_queue_size'] = int(kwargs[current_op_type_queue])

            if len(temp_data['children']) > 0:
                temp_data = temp_data['children'][0]

        start_time = time.time()
        deserialized_dataset = ds.deserialize(input_dict=origin_data)

        for _ in deserialized_dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            data_nums = data_nums + 1
            if data_nums > 100:
                break

        end_time = time.time()
        cost_time = start_time - end_time
        return cost_time, deserialized_dataset

    def autotune_optimize(self, **kwargs):
        cost_time, _ = self.internal_autotune_optimize(kwargs, False)
        return cost_time

    def optimize(self, params, init_points, n_iter, save_path=None):
        # 获取op type信息
        count = 1
        origin_data = self.process_data
        while ('children' in self.process_data):
            op_type_key = str(self.process_data['op_type']) + str(count)
            op_queue_key = op_type_key + "_Queue"
            self.op_type_keys.append(op_type_key)
            self.op_type_keys.append(op_queue_key)
            # print("op_type" + str(count) + ": " + str(self.process_data['op_type']))
            if len(self.process_data['children']) > 0:
                count = count + 1
                self.process_data = self.process_data['children'][0]
            else:
                break
        self.process_data = origin_data
        search_workers = params
        search_params = {}
        for key in self.op_type_keys:
            search_params[key] = search_workers

        optimizer = BayesianOptimization(
        f=self.autotune_optimize,
        pbounds=search_params
                )

        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        # save params
        print(optimizer.max)
        cost_time, dataset = self.internal_autotune_optimize(optimizer.max["params"], True)
        params_save_path = save_path if save_path else "./best_optimize.json"
        ds.serialize(dataset, params_save_path)
        return optimizer.max