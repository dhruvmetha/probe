from params_proto import PrefixProto

class RunCfg(PrefixProto):
    class runs:
        device = 'cuda:0'
        mode = 'train' # 'train', 'sim_test', 'real_test'
        model_name = 'transformer' # 'transformer', 'velocity_model'
        save_root = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor'
        log_folder = 'results'
        experiments_folder = 'experiments_14_mar_same_policy/2_obs'
        real_experiments_folder = 'real_experiments/13_mar'
        class train:
            # data_source = [
            #     '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19/movable_1_obs/balanced/train_0.pkl', 
            #     '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19/movable_1_obs/balanced/train_1.pkl', 
            #     '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19/static_1_obs/balanced/train_0.pkl', 
            #     '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19/static_1_obs/balanced/train_1.pkl'
            # ]

            data_source = [
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb21/2_obs/balanced/train_0.pkl', 
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb21/2_obs/balanced/train_1.pkl',
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb21/2_obs/balanced/train_2.pkl',
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb22/2_obs/balanced/train_0.pkl', 
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb22/2_obs/balanced/train_imm.pkl',
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb22/2_obs/balanced/train_mv.pkl',
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb22/2_obs/balanced/train_2.pkl',
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_mar14/3_obs/no_balanced/train_3.pkl'
            ]

            inputs = ['joint_pos', 'joint_vel', 'torques', 'pose']
            outputs = ['confidence', 'contact', 'movable', 'pose', 'size']
            # inputs = ['joint_pos', 'joint_vel', 'torques']
            # outputs = ['velocity']

            save_directory = '2_obs'
        
        class test:
            # data_source = [
            #     ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/static_1_obs/balanced/train_imm.pkl'], 
            #     ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/movable_1_obs/balanced/train_mv.pkl']
            #     ]
            
            data_source = [
                # [
                #     '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb23_same_policy/2_obs/balanced/train_imm.pkl',
                # ],
                # [
                #     '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb23_same_policy/2_obs/balanced/train_mv.pkl',
                # ],
                [
                    '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb23_same_policy/2_obs/balanced/train_2.pkl'
                ]

                # ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_mv.pkl'], 
                # ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_2.pkl']
            ]  * 5

            log_folder = 'losses'
            
            inputs = [
                ['joint_pos', 'joint_vel', 'torques', 'pose'],
                ['joint_pos', 'joint_vel', 'torques'],
                ['joint_pos', 'joint_vel'],
                ['joint_pos'],
                ['torques', 'pose']
            ]
            outputs = [['confidence', 'movable', 'pose', 'size']] * 5
            ckpt = [
                # '2_obs/q_to_cdctmvposesize/2024-02-22_20-48-49/checkpoints/transformer_weights_9.pt',
                '2_obs/qqdtaupose_to_cdmvposesize/2024-03-13_01-22-25/checkpoints/transformer_weights_12.pt',
                '2_obs/qqdtau_to_cdmvposesize/2024-03-13_01-27-16/checkpoints/transformer_weights_12.pt',
                '2_obs/qqd_to_cdmvposesize/2024-03-13_01-27-48/checkpoints/transformer_weights_12.pt',
                '2_obs/q_to_cdmvposesize/2024-03-13_01-28-25/checkpoints/transformer_weights_12.pt',
                '2_obs/taupose_to_cdmvposesize/2024-03-13_01-29-04/checkpoints/transformer_weights_12.pt',
                # '2_obs/qqdtaupose_to_cdctmvposesize/2024-02-23_23-11-16/checkpoints/transformer_weights_10.pt',
                # '2_obs/pose_to_cdctmvposesize/2024-02-22_20-57-43/checkpoints/transformer_weights_9.pt',
            ]
            experiment_name = ['qqdtaupose_to_cdmvposesize', 'qqdtau_to_cdmvposesize', 'qqd_to_cdmvposesize', 'q_to_cdmvposesize', 'taupose_to_cdmvposesize']

            # inputs = [['pose'], ['pose'], ['pose']]
            # outputs = [['pose'], ['pose'], ['pose']]
            # ckpt = [
            #     'p_from_pose/2024-02-20_11-13-20/checkpoints/transformer_weights_9.pt',
            #     'p_from_pose/2024-02-20_11-13-20/checkpoints/transformer_weights_9.pt',
            #     'p_from_pose/2024-02-20_11-13-20/checkpoints/transformer_weights_9.pt'
            #     ]
            # experiment_name = ['p_from_pose_static', 'p_from_pose_movable', 'p_from_pose_both']

        class real_test:
            root_folder = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/real_robot_data'
            # sub_folders = ['sep14', 'sep15', 'feb22_feb25']

            # sub_folders = ['mar3/feb22', 'mar3/feb21',]  #'sep14', 'sep15' #'feb27/feb21', 'feb27/feb22', 'feb22_feb26']# 'sep14', 'sep15', 
            # log_folder = '8_mar_others/2024-03-09_10-14-42'

            # sub_folders = ['3_mar_easy_static', ]  #'sep14', 'sep15' #'feb27/feb21', 'feb27/feb22', 'feb22_feb26']# 'sep14', 'sep15', 
            # log_folder = '3_mar_easy_static/2024-03-09_10-14-42'
            
            # movable
            # sub_folders = ['3_mar_easy_movable', '8_mar_easy_movable/aug26', '8_mar_easy_movable/feb22', '8_mar_easy_movable/feb23']
            # log_folder = '9_mar/all_easy_movable/2024-03-07_22-01-46'

            # static
            # sub_folders = ['3_mar_easy_static', '8_mar_easy_static/aug26', '8_mar_easy_static/aug30', '8_mar_easy_static/feb21', '8_mar_easy_static/feb22']
            # log_folder = '9_mar/all_easy_static/2024-03-07_22-01-46'

            

            # # medium
            sub_folders = ['3_mar_medium', '8_mar_medium/aug26', '8_mar_medium/aug30', '8_mar_medium/feb21', '8_mar_medium/feb22']
            log_folder = '9_mar/all_medium/2024-03-14_18-50-22'

            inputs = ['joint_pos', 'joint_vel', 'torques', 'pose']
            outputs = ['confidence', 'contact', 'movable', 'pose', 'size']
            save_directory = ''
            
            # ckpt = '2_obs/qqdtaupose_to_cdctmvposesize/2024-03-07_22-01-46/checkpoints/transformer_weights_10.pt'
            ckpt = '2_obs/qqdtaupose_to_cdctmvposesize/2024-03-14_18-50-22/checkpoints/transformer_weights_3.pt'
            experiment_name = 'full_prediction_real'

    class transformer:
        class model_params:
            sequence_length = 1500
            hidden_state_size = 1024
            embed_size = 512
            num_heads = 2
            num_layers = 4

        class data_params:
            obstacles = 2
            inputs = {
                    'joint_pos' : 12, 
                    'joint_vel': 12, 
                    'torques': 12, 
                    'projected_gravity': 3, 
                    'pose': 3,
                    'velocity': 2
                    }
            outputs = {
                    'confidence': 1,
                    'contact': 1, 
                    'movable': 1, 
                    'pose': 3, 
                    'size': 2
            }
            
        
        class train_params:
            learning_rate = 1e-4
            epochs = 20
            train_batch_size = 16
            val_batch_size = 16
            test_batch_size = 1
            train_test_split = 0.98

        class loss_scales:
            confidence_scale = 1
            contact_scale = 1/3
            movable_scale = 1
            pos_scale = 5
            yaw_scale = 2 * 10
            size_scale = 3

        class logging:
            eval_every = 200
            save_every = 500
            print_every = 50
            test_every = eval_every
            animation = True

    class velocity_model:
        class model_params:
            sequence_length = 1500
            hidden_state_size = 64
            embed_size = 64
            num_heads = 2
            num_layers = 2

        class data_params:
            inputs = {
                    'joint_pos' : 12, 
                    'joint_vel': 12, 
                    'torques': 12, 
                    'projected_gravity': 3, 
                    'pose': 3
                    }
            outputs = {
                    'velocity': 2
                    }
        
        class train_params:
            learning_rate = 1e-4
            epochs = 5
            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 1
            train_test_split = 0.95

        class loss_scales:
            velocity_scale = 1

        class logging:
            eval_every = 200
            save_every = 500
            print_every = 50
            test_every = eval_every
            animation = True

    