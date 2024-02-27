from params_proto import PrefixProto

class RunCfg(PrefixProto):
    class runs:
        device = 'cuda:0'
        mode = 'train' # 'train', 'sim_test', 'real_test'
        model_name = 'transformer' # 'transformer', 'velocity_model'
        save_root = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor'
        log_folder = 'results'
        experiments_folder = 'experiments_feb23_new_policy'
        real_experiments_folder = 'real_experiments'
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
                '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb22/2_obs/balanced/train_2.pkl'
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
                [
                    '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb23_new_policy/2_obs/balanced/train_0.pkl', 
                    '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb23_new_policy/2_obs/balanced/train_1.pkl',
                    '/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb23_new_policy/2_obs/balanced/train_2.pkl'
                ]
                # ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_mv.pkl'], 
                # ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_2.pkl']
            ]  * 5

            log_folder = 'losses'
            
            # inputs = [['joint_pos', 'joint_vel', 'torques'], ['joint_pos', 'joint_vel', 'torques'], ['joint_pos', 'joint_vel', 'torques']]
            # outputs = [['pose', 'size'], ['pose', 'size'], ['pose', 'size']]
            # ckpt = [
            #     'ps_from_qqdottau/2024-02-20_07-54-16/checkpoints/transformer_weights_9.pt',
            #     'ps_from_qqdottau/2024-02-20_07-54-16/checkpoints/transformer_weights_9.pt',
            #     'ps_from_qqdottau/2024-02-20_07-54-16/checkpoints/transformer_weights_9.pt'
            #     ]
            # experiment_name = ['ps_from_qqdottau_static', 'ps_from_qqdottau_movable', 'ps_from_qqdottau_both']

            # inputs = [['joint_pos', 'joint_vel', 'torques', 'pose'], ['joint_pos', 'joint_vel', 'torques', 'pose'], ['joint_pos', 'joint_vel', 'torques', 'pose']]
            # outputs = [['pose', 'size'], ['pose', 'size'], ['pose', 'size']]
            # ckpt = [
            #     'ps_from_qqdottaupose/2024-02-20_07-53-42/checkpoints/transformer_weights_9.pt',
            #     'ps_from_qqdottaupose/2024-02-20_07-53-42/checkpoints/transformer_weights_9.pt',
            #     'ps_from_qqdottaupose/2024-02-20_07-53-42/checkpoints/transformer_weights_9.pt'
            #     ]
            # experiment_name = ['ps_from_qqdottaupose_static', 'ps_from_qqdottaupose_movable', 'ps_from_qqdottaupose_both']

        
            # inputs = [['joint_pos', 'joint_vel', 'torques', 'pose'], ['joint_pos', 'joint_vel', 'torques', 'pose'], ['joint_pos', 'joint_vel', 'torques', 'pose']]
            # outputs = [['pose'], ['pose'], ['pose']]
            # ckpt = [
            #     'p_from_qqdottaupose/2024-02-20_07-55-23/checkpoints/transformer_weights_9.pt',
            #     'p_from_qqdottaupose/2024-02-20_07-55-23/checkpoints/transformer_weights_9.pt',
            #     'p_from_qqdottaupose/2024-02-20_07-55-23/checkpoints/transformer_weights_9.pt'
            #     ]
            # experiment_name = ['p_from_qqdottaupose_static', 'p_from_qqdottaupose_movable', 'p_from_qqdottaupose_both']

            # inputs = [['joint_pos', 'joint_vel', 'torques'], ['joint_pos', 'joint_vel', 'torques'], ['joint_pos', 'joint_vel', 'torques']]
            # outputs = [['pose'], ['pose'], ['pose']]
            # ckpt = [
            #     'p_from_qqdottau/2024-02-20_07-55-06/checkpoints/transformer_weights_9.pt',
            #     'p_from_qqdottau/2024-02-20_07-55-06/checkpoints/transformer_weights_9.pt',
            #     'p_from_qqdottau/2024-02-20_07-55-06/checkpoints/transformer_weights_9.pt'
            #     ]
            # experiment_name = ['p_from_qqdottau_static', 
            # 'p_from_qqdottau_movable', 'p_from_qqdottau_both']

            # inputs = [['joint_pos', 'joint_vel'], ['joint_pos', 'joint_vel'], ['joint_pos', 'joint_vel']]
            # outputs = [['pose'], ['pose'], ['pose']]
            # ckpt = [
            #     'p_from_qqdot/2024-02-20_07-56-15/checkpoints/transformer_weights_9.pt',
            #     'p_from_qqdot/2024-02-20_07-56-15/checkpoints/transformer_weights_9.pt',
            #     'p_from_qqdot/2024-02-20_07-56-15/checkpoints/transformer_weights_9.pt'
            #     ]
            # experiment_name = ['p_from_qqdot_static', 'p_from_qqdot_movable', 'p_from_qqdot_both']

            inputs = [['joint_pos'], ['joint_pos', 'joint_vel'], ['joint_pos', 'joint_vel', 'torques'], ['joint_pos', 'joint_vel', 'torques', 'pose'], ['pose']]
            outputs = [['confidence', 'contact', 'movable', 'pose', 'size']] * 5 
            ckpt = [
                '2_obs/q_to_cdctmvposesize/2024-02-22_20-48-49/checkpoints/transformer_weights_9.pt',
                '2_obs/qqd_to_cdctmvposesize/2024-02-22_20-54-47/checkpoints/transformer_weights_9.pt',
                '2_obs/qqdtau_to_cdctmvposesize/2024-02-22_20-56-43/checkpoints/transformer_weights_9.pt',
                '2_obs/qqdtaupose_to_cdctmvposesize/2024-02-22_20-56-53/checkpoints/transformer_weights_9.pt',
                '2_obs/pose_to_cdctmvposesize/2024-02-22_20-57-43/checkpoints/transformer_weights_9.pt',
            ]
            experiment_name = ['all_from_q', 'all_from_qqdot', 'all_from_qqdottau', 'all_from_qqdottaupose', 'all_from_pose']

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
            sub_folders = ['feb22_feb26', 'sep14', 'sep15']

            inputs = ['joint_pos', 'joint_vel', 'torques', 'pose']
            outputs = ['confidence', 'contact', 'movable', 'pose', 'size']

            log_folder = '2_obs/full_prediction/2024-02-22_04-18-38'
            save_directory = ''
            # ckpt = '2_obs/qqdtau_to_cdctmvposesize/2024-02-22_20-56-43/checkpoints/transformer_weights_15.pt'
            ckpt = '2_obs/qqdtaupose_to_cdctmvposesize/2024-02-26_18-19-49/checkpoints/transformer_weights_5.pt'
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
                    'pose': 3
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
            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 1
            train_test_split = 0.95

        class loss_scales:
            confidence_scale = 1
            contact_scale = 1/3
            movable_scale = 1
            pos_scale = 5
            yaw_scale = 2 * 10
            size_scale = 2

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

    