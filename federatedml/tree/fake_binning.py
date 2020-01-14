from arch.api.table.eggroll.table_impl import DTable
from federatedml.tree.test.homo_local_testing import dtable
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.param.feature_binning_param import FeatureBinningParam
import numpy as np
from federatedml.feature.instance import Instance
import functools

class FakeBinning():

    def __init__(self,bin_num=10):
        self.bin_num = bin_num

    def helper(self,instance,split_points):
        feat = instance.features
        sparse_feat = []
        for idx, v in enumerate(feat):
            sparse_feat.append(np.argmax(split_points[idx] > v) - 1)
        return Instance(inst_id=instance.inst_id,features=sparse_feat,label=instance.label)

    def fit(self,Dtable:DTable):
        arr = []
        for row in Dtable.collect():
            arr.append(row[1].features)

        arr = np.stack(arr)
        split_points = []
        width = arr.shape[1]
        for num in range(width):
            col_max = arr[:,num].max()
            col_min = arr[:,num].min()
            split_points.append(np.arange(col_min,col_max,(col_max-col_min)/self.bin_num))

        self.split_points = np.stack(split_points)

        func = functools.partial(self.helper,split_points=self.split_points)
        new_table = table.mapValues(func)
        return new_table,self.split_points,{k:0 for k in range(self.bin_num)}


if __name__ == '__main__':
    fake_binning = FakeBinning()
    table = dtable
    binning = QuantileBinning(params=FeatureBinningParam(bin_num=10))
    binning.fit_split_points(table)
    a,b,c = binning.convert_feature_to_bin(table)
    e,f,g = fake_binning.fit(table)