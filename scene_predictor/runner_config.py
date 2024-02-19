from params_proto import PrefixProto

class RunCfg(PrefixProto):

    class runs:
        device = 'cuda:0'
        train_mode = False
        model_name = 'transformer'
        save_root = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results'
        class train:
            data_source = ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_0/balanced/train_0.pkl', '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_0/balanced/train_1.pkl', '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_0/balanced/train_2.pkl']

            inputs = ['joint_pos', 'joint_vel', 'torques']
            outputs = ['pose']
        
        class test:
            data_source = [['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_0/balanced/train_1.pkl']]
            inputs = [['joint_pos', 'joint_vel', 'torques']]
            outputs = [['pose']]
            ckpt = ['/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results/transformer/2024-02-19_13-28-55/checkpoints/transformer_weights_4.pt']
        
    class transformer:
        class model_params:
            sequence_length = 1500
            hidden_state_size = 1024
            embed_size = 512
            num_heads = 2
            num_layers = 2

        class data_params:
            obstacles = 2
            inputs = {
                    'joint_pos' : 12, 
                    'joint_vel': 12, 
                    'torques': 12, 
                    'projected_gravity': 3, 
                    'pose': 3
                    }
            outputs = {
                    'contact': 1, 
                    'movable': 1, 
                    'pose': 3, 
                    'size': 2
            }
            
        
        class train_params:
            learning_rate = 1e-4
            epochs = 50
            train_batch_size = 64
            val_batch_size = 64
            test_batch_size = 1
            train_test_split = 0.95

        class loss_scales:
            contact_scale = 1/3
            movable_scale = 1/2
            pos_scale = 2
            yaw_scale = 2 * 10
            size_scale = 2

        class logging:
            eval_every = 200
            save_every = 500
            print_every = 50
            test_every = eval_every
            animation = True