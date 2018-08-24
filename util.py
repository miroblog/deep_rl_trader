import numpy as np
from rl.core import Processor
from rl.util import WhiteningNormalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ADDITIONAL_STATE = 4
class NormalizerProcessor(Processor):
    def __init__(self):
        self.scaler = StandardScaler()
        self.normalizer = None

    def process_state_batch(self, batch):
        batch_len = batch.shape[0]
        k = []
        for i in range(batch_len):
            observe = batch[i][..., :-ADDITIONAL_STATE]
            observe = self.scaler.fit_transform(observe)
            agent_state = batch[i][..., -ADDITIONAL_STATE:]
            temp = np.concatenate((observe, agent_state),axis=1)
            temp = temp.reshape((1,) + temp.shape)
            k.append(temp)
        batch = np.concatenate(tuple(k))
        return batch
