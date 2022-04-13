from multimodal_transformers.model.tabular_modeling_auto import AutoModelWithTabular
from transformers import AutoTokenizer, AutoConfig

class Longformer():
    '''
    Contains pipeline for inference with the Longformer
    '''
    def __init__(self, path, *args):
        super(Longformer, self).__init__(*args)
        self.model = AutoModelWithTabular.from_pretrained(path)

class HybridLongformer():
    '''
    Contains pipeline for inference with the tabular Longformer
    '''
    def __init__(self, path, *args):
        super(HybridLongformer, self).__init__(*args)
        self.model = AutoModelWithTabular.from_pretrained(path)

class LSTM():
    '''
    Contains pipeline for inference with the LSTM
    '''
    def __init__(self, path, *args):
        super(LSTM, self).__init__(*args)
        self.model = AutoModelWithTabular.from_pretrained(path)