from training_arch.transformer_model import TransformerModel

ARCHS = {
    'transformer': TransformerModel,
}

class Runner:
    def __init__(self, model, data_source, device='cuda:0'):
        self.model = ARCHS[model.arch](data_source, device)

    def learn(self):
        self.model.train()



    

