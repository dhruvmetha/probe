from training_arch.transformer_model import TransformerModel
from params_proto import PrefixProto

import os
import json

ARCHS = {
    'transformer': TransformerModel,
}
class Runner:
    def __init__(self, train_cfg, model_name, data_source, save_folder, device='cuda:0'):
        
        self.model = ARCHS[model_name](train_cfg, data_source, save_folder, device)

    def learn(self):
        self.model.train()

    def load_model(self, ckpt):
        self.model.load_model(ckpt)

    def test(self, test_data_source, inputs, outputs, filename):
        recorded_loss = self.model.test(test_data_source)
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'recorded_loss': recorded_loss
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

if __name__ == '__main__':
    from runner_config import RunCfg
    
    runs = RunCfg.runs
    
    model_name  = runs.model_name
    train_cfg = RunCfg.transformer # eval(f"RunCfg.{model_name}")
    train_data_source = runs.train.data_source
    test_data_source = runs.test.data_source
    device = runs.device
    save_root = runs.save_root
    save_folder = f'{save_root}/{model_name}'

    if runs.train_mode:
        inputs = runs.train.inputs
        outputs = runs.train.outputs

        input_params = train_cfg.data_params.inputs
        output_params = train_cfg.data_params.outputs
        new_input_params = {}
        for i in inputs:
            new_input_params[i] = input_params[i]
        new_output_params = {}
        for o in outputs:
            new_output_params[o] = output_params[o]

        train_cfg.data_params.inputs = new_input_params
        train_cfg.data_params.outputs = new_output_params
        train_cfg.logging.animation = False
        train_cfg.train_params.epochs = 5

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        runner = Runner(train_cfg, model_name, train_data_source, save_folder, 'cuda:0')
        runner.learn()
    else:
       for idx in range(len(runs.test.inputs)):
            inputs = runs.test.inputs[idx]
            outputs = runs.test.outputs[idx]
            input_params = train_cfg.data_params.inputs
            output_params = train_cfg.data_params.outputs
            new_input_params = {}
            for i in inputs:
                new_input_params[i] = input_params[i]
            new_output_params = {}
            for o in outputs:
                new_output_params[o] = output_params[o]
            train_cfg.data_params.inputs = new_input_params
            train_cfg.data_params.outputs = new_output_params
           
            runner = Runner(train_cfg, model_name, None, save_folder, 'cuda:0')
            runner.load_model(runs.test.ckpt[idx])
            runner.test(test_data_source[idx], inputs, outputs, f'losses_{idx}.json')

    # filename = 'losses_1.json'
    # runner.test(test_data_source, inputs, outputs, filename)
