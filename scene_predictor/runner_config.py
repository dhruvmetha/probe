from params_proto import PrefixProto

class RunCfg(PrefixProto):

    class runs:
        device = 'cuda:0'
        mode = 'train' # 'train', 'sim_test', 'real_test'
        model_name = 'transformer'
        save_root = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor'
        log_folder = 'results'
        experiments_folder = 'experiments_on_contact'
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

            save_directory = '2_obs'
        
        class test:
            # data_source = [
            #     ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/static_1_obs/balanced/train_imm.pkl'], 
            #     ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/movable_1_obs/balanced/train_mv.pkl']
            #     ]
            
            data_source = [
                ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_imm.pkl'], 
                ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_mv.pkl'], 
                ['/common/users/dm1487/legged_manipulation_data_store/trajectories/iros24_play_feb19_test/2_obs/balanced/train_2.pkl']]

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

            inputs = [['joint_pos'], ['joint_pos'], ['joint_pos']]
            outputs = [['pose'], ['pose'], ['pose']]
            ckpt = [
                'p_from_q/2024-02-20_07-56-29/checkpoints/transformer_weights_9.pt',
                'p_from_q/2024-02-20_07-56-29/checkpoints/transformer_weights_9.pt',
                'p_from_q/2024-02-20_07-56-29/checkpoints/transformer_weights_9.pt'
                ]
            experiment_name = ['p_from_q_static', 'p_from_q_movable', 'p_from_q_both']

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
            sub_folders = ['sep14', 'sep15']

            inputs = ['joint_pos', 'joint_vel', 'torques', 'pose']
            outputs = ['confidence', 'contact', 'movable', 'pose', 'size']

            log_folder = '2_obs/full_prediction/2024-02-22_04-18-38'
            save_directory = ''
            ckpt = '2_obs/full_prediction/2024-02-22_04-18-38/checkpoints/transformer_weights_10.pt'
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
            epochs = 50
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