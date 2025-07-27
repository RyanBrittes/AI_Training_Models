
class FeatureCrossing():
    def get_cross_feature(self, dataset, col_01, col_02):
        dataset[col_01 + '_' + col_02] = dataset[col_01] + '_' + dataset[col_02]
        return dataset