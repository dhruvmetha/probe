from training_arch.transformer_model import TransformerModel
from params_proto import PrefixProto

import os
import json

ARCHS = {
    'transformer': TransformerModel,
}
class Runner:
    def __init__(self, train_cfg, model_name, data_source, save_folder, directory, device='cuda:0'):
        
        self.model = ARCHS[model_name](train_cfg, data_source, save_folder, directory, device)

    def learn(self):
        self.model.train()

    def load_model(self, ckpt):
        self.model.load_model(ckpt)

    def test(self, test_data_source, filename, inputs, outputs, ckpt):
        recorded_loss = self.model.test(test_data_source)
        data = {
            'inputs': inputs,
            'outputs': outputs,
            'ckpt': ckpt,
            'recorded_loss': recorded_loss
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def test_real(self, test_data_source, log_folder, results_json, inputs, outputs, ckpt):
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        recorded_loss = self.model.test_real(test_data_source, log_folder)
        # data = {
        #     'inputs': inputs,
        #     'outputs': outputs,
        #     'ckpt': ckpt,
        #     'recorded_loss': recorded_loss
        # }
        # with open(f'{log_folder}/{results_json}', 'w') as f:
        #     json.dump(data, f)

if __name__ == '__main__':
    from runner_config import RunCfg
    from glob import glob
    runs = RunCfg.runs
    
    model_name  = runs.model_name
    train_cfg = RunCfg.transformer # eval(f"RunCfg.{model_name}")
    train_data_source = runs.train.data_source
    test_data_source = runs.test.data_source
    real_data_source = runs.real_test.root_folder
    device = runs.device
    save_root = runs.save_root
    log_folder = runs.log_folder
    experiments_folder = runs.experiments_folder
    save_folder = f'{save_root}/{log_folder}/{model_name}'
    experiments_folder = f'{save_root}/{experiments_folder}/{model_name}'
    real_experiment_folder = f'{save_root}/{runs.real_experiments_folder}/{model_name}'

    if runs.mode == 'train':
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
        # train_cfg.train_params.epochs = 10

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        directory = runs.train.save_directory
    
        runner = Runner(train_cfg, model_name, train_data_source, save_folder, directory, 'cuda:0')
        runner.learn()
    elif runs.mode == 'sim_test':
        if not os.path.exists(experiments_folder):
            os.makedirs(experiments_folder)

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
            ckpt_file = os.path.join(save_folder, runs.test.ckpt[idx])

            runner.load_model(ckpt_file)
            test_args = {
                'inputs': inputs,
                'outputs': outputs,
                'ckpt': runs.test.ckpt[idx] 
            }
            runner.test(test_data_source[idx], f'{experiments_folder}/{runs.test.experiment_name[idx]}.json', **test_args)

        # create plots for all.
    
    elif runs.mode  == 'real_test':
        if not os.path.exists(real_experiment_folder):
            os.makedirs(real_experiment_folder)
        
        files = []
        for sf in runs.real_test.sub_folders:
            for i in os.listdir(f'{real_data_source}/{sf}'):
                # path = os.path.join(f'{real_data_source}/{sf}', i)
                print(f'{real_data_source}/{sf}/{i}/*.npz')
                files.append(sorted(glob(f'{real_data_source}/{sf}/{i}/*.npz'))[-1])

        inputs = runs.real_test.inputs
        outputs = runs.real_test.outputs
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
        ckpt_file = os.path.join(save_folder, runs.real_test.ckpt)

        runner.load_model(ckpt_file)
        
        test_args = {
            'inputs': runs.real_test.inputs,
            'outputs': runs.real_test.outputs,
            'ckpt': runs.real_test.ckpt 
        }

        runner.test_real(files, f'{real_experiment_folder}', f'{runs.test.experiment_name}.json', **test_args)
        

        

    # filename = 'losses_1.json'
    # runner.test(test_data_source, inputs, outputs, filename)
