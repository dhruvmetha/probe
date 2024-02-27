import numpy as np

class world_cfg:
    CUSTOM_BLOCK = True
    class movable_block:
        name = 'movable_block'
        size_x_range = [0.1, 0.4]
        size_y_range = [1.0, 1.75] # [0.8, 1.5]
        pos_x_range = [.8, 1.5]
        pos_y_range = [-0.1, 0.1]
        block_density_range = [1, 6]

    class fixed_block:
        add_to_obs = True
        name = 'fixed_block'
        num_obs = 2
        size_x_range = [0.1, 0.4]
        size_y_range = [0.3, 0.8] # [0.5, 0.6] # [0.8, 1.5]
        pos_x_range = [1.8, 1.95]
        pos_y_range = [-0.5, 0.5]

RANDOM_INPLAY_ASSETS = [
        # {
        #     'name': ['fb_three_fixed_block_2', 'fb_three_fixed_block_1', 'fb_three_movable_block_1'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.8, 1.3]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },

        # {
        #     'name': ['fb_three_fixed_block_4', 'fb_three_fixed_block_3', 'fb_three_movable_block_2'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.8, 1.3]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },

        # {
        #     'name': ['fb_three_fixed_block_6', 'fb_three_fixed_block_5', 'fb_three_movable_block_3'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },

        # {
        #     'name': ['fb_three_fixed_block_8', 'fb_three_fixed_block_7', 'fb_three_movable_block_4'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.8, 1.3]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.8, 1.3]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },

        {
            'name': ['bb_three_fixed_block_2', 'bb_three_fixed_block_1', 'bb_three_movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['bb_three_fixed_block_4', 'bb_three_fixed_block_3', 'bb_three_movable_block_2'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.4, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
        
        {
            'name': ['bb_three_fixed_block_6', 'bb_three_fixed_block_5', 'bb_three_movable_block_3'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
        # {
        #     'name': ['bb_three_fixed_block_8', 'bb_three_fixed_block_7', 'bb_three_movable_block_4'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.8, 1.3]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },
    ]

EVAL_INPLAY_ASSETS = [

        {
            'name': ['fixed_asset_1', 'fixed_asset_2', 'movable_asset_1'],
            'size': [[lambda:0.3, lambda:0.8, lambda: 0.4], [lambda:0.3, lambda:0.4, lambda: 0.4], [lambda:0.3, lambda:0.9, lambda: 0.4]],
            'pos': [[1.5, 0.5, 0.2], [2.25, -0.15, 0.2], [1.5, -0.4, 0.2]],
            'density': [10000, 10000, 8]
        },
        
        {
            'name': ['fixed_asset_3', 'fixed_asset_4', 'movable_asset_2'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.4, lambda: 0.4], [lambda: 0.3, lambda: 0.9, lambda: 0.4]],
            'pos': [[1.5, 0.5, 0.2], [2.25, -0.6, 0.2], [1.5, -0.4, 0.2]],
            'density': [10000, 10000, 8]
        },

        {
            'name': ['fixed_asset_6', 'fixed_asset_7', 'movable_asset_4'],
            'size': [[lambda:0.3, lambda:0.8, lambda: 0.4], [lambda:0.3, lambda:0.4, lambda: 0.4], [lambda:0.3, lambda:0.9, lambda: 0.4]],
            'pos': [[1.5, -0.5, 0.2], [2.25, 0.15, 0.2], [1.5, 0.4, 0.2]],
            'density': [10000, 10000, 8]
        },

        {
            'name': ['fixed_asset_8', 'fixed_asset_9', 'movable_asset_5'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.4, lambda: 0.4], [lambda: 0.3, lambda: 0.9, lambda: 0.4]],
            'pos': [[1.5, -0.5, 0.2], [2.25, 0.6, 0.2], [1.5, 0.4, 0.2]],
            'density': [10000, 10000, 8]
        },
        
        {
            'name': ['fixed_asset_10',  'movable_asset_6'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.9, lambda: 0.4]],
            'pos': [[1.5, -0.5, 0.2], [1.3, 0.4, 0.2]],
            'density': [10000, 8]
        },

        # {
        #     'name': ['fixed_asset_11', 'fixed_asset_12', 'movable_asset_7'],
        #     'size':  [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.6, lambda: 0.4]],
        #     'pos': [[1.3, 0.5, 0.2], [2.1, -0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_13', 'fixed_asset_14', 'movable_asset_8'],
        #     'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.6, lambda: 0.4]],
        #     'pos': [[1.3, -0.5, 0.2], [2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_11_dup_1', 'fixed_asset_12_dup_1', 'movable_asset_7_dup_1'],
        #     'size':  [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
        #     'pos': [[1.3, 0.5, 0.2], [2.1, -0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_13_dup_1', 'fixed_asset_14_dup_1', 'movable_asset_8_dup_1'],
        #     'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.7, lambda: 0.4]],
        #     'pos': [[1.3, -0.5, 0.2], [2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_11_dup_2', 'fixed_asset_12_dup_2', 'movable_asset_7_dup_2'],
        #     'size':  [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
        #     'pos': [[1.3, 0.5, 0.2], [2.1, -0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_13_dup_2', 'fixed_asset_14_dup_2', 'movable_asset_8_dup_2'],
        #     'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.7, lambda: 0.4]],
        #     'pos': [[1.3, -0.5, 0.2], [2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_15', 'movable_asset_9'],
        #     'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
        #     'pos': [[1.3, -0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_16', 'movable_asset_10'],
        #     'size': [[lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
        #     'pos': [[2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 8]
        # },

        {
            'name': ['fixed_block_1', 'movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.3, 1.7]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 8][::-1]
        }

        # {
        #     'name': ['fixed_block_1', 'movable_block_1'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.5]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 8][::-1]
        # }
    ]

# EVAL_INPLAY_ASSETS = [*INPLAY_ASSETS]
# INPLAY_ASSETS = [*EVAL_INPLAY_ASSETS]

TASK_ONE = [
        {
            'name': ['fixed_block_3'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000][::-1]
        },

        # {
        #     'name': ['fixed_block_4'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000][::-1]
        # },

        # {
        #     'name': ['fixed_block_5'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000][::-1]
        # },

        {
            'name': ['movable_block_3'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.7, 1.5]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
            'density': [8][::-1]
        },
]

TASK_TWO = [
        # {
        #     'name': ['fixed_block_1', 'movable_block_1'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.5, 0.5]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.3, 1.3]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 8][::-1]
        # },

        {
            'name': ['fixed_block_1', 'movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 1.5]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 8][::-1]
        },

        {
            'name': ['fixed_block_2', 'movable_block_2'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.0, 1.5]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 8][::-1]
        },

        # {
        #     'name': ['fixed_block_3'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000][::-1]
        # },

        # {
        #     'name': ['movable_block_3'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.7, 1.5]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [8][::-1]
        # },

        # {
        #     'name': ['movable_block_4'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.2, 1.8]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [8][::-1]
        # }
]

TASK_THREE = [
        {
            'name': ['bb_three_fixed_block_2', 'bb_three_fixed_block_1', 'bb_three_movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['bb_three_fixed_block_4', 'bb_three_fixed_block_3', 'bb_three_movable_block_2'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.4, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
        
        {
            'name': ['bb_three_fixed_block_6', 'bb_three_fixed_block_5', 'bb_three_movable_block_3'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
]

# TASK_REAL = [ 
#         {
#                 'name': ['custom_movable_1'][::-1],
#                 'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.2, 1.2]), 2), lambda: 0.4]][::-1], 
#                 'pos': [[1.6363, 0.0382, 0.2]][::-1],
#                 'density': [8][::-1]
#         },
# ]   

TASK_REAL = [ 
        {
                'name': ['custom_fixed_1', ' custom_movable_1'][::-1],
                'size': [[lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[0.6, 0.8]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.4, 0.4]), 2), lambda: round(np.random.uniform(*[1.5, 1.5]), 2), lambda: 0.4]][::-1], 
                'pos': [[1.6, -0.4, 0.2], [1.0, -0.0382, 0.2]][::-1],
                'density': [10000, 8][::-1]
        },
]   


##
    ##### 
####
TASK_1 = [
        

        {
            'name': ['fixed_asset_1', 'fixed_asset_2', 'movable_asset_1'][::-1],
            'size': [[lambda:0.4, lambda:0.8, lambda: 0.4], [lambda:0.4, lambda:0.4, lambda: 0.4], [lambda:0.4, lambda:0.9, lambda: 0.4]][::-1],
            'pos': [[1.5, 0.5, 0.2], [2.25, -0.15, 0.2], [1.0, -0.4, 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
        
        {
            'name': ['fixed_asset_3', 'fixed_asset_4', 'movable_asset_2'][::-1],
            'size': [[lambda: 0.4, lambda: 0.8, lambda: 0.4], [lambda: 0.4, lambda: 0.4, lambda: 0.4], [lambda: 0.4, lambda: 0.9, lambda: 0.4]][::-1],
            'pos': [[1.5, 0.5, 0.2], [2.25, -0.6, 0.2], [1.0, -0.4, 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['fixed_asset_6', 'fixed_asset_7', 'movable_asset_4'][::-1],
            'size': [[lambda:0.4, lambda:0.8, lambda: 0.4], [lambda:0.4, lambda:0.4, lambda: 0.4], [lambda:0.4, lambda:0.9, lambda: 0.4]][::-1],
            'pos': [[1.5, -0.5, 0.2], [2.25, 0.15, 0.2], [1.0, 0.4, 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['fixed_asset_8', 'fixed_asset_9', 'movable_asset_5'][::-1],
            'size': [[lambda: 0.4, lambda: 0.8, lambda: 0.4], [lambda: 0.4, lambda: 0.4, lambda: 0.4], [lambda: 0.4, lambda: 0.9, lambda: 0.4]][::-1],
            'pos': [[1.5, -0.5, 0.2], [2.25, 0.6, 0.2], [1.0, 0.4, 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
]

#####
      #####

###########
TASK_2 = [



        {
            'name': ['fixed_asset_11', 'fixed_asset_12', 'movable_asset_7'][::-1],
            'size':  [[lambda: 0.4, lambda: 0.6, lambda: 0.4], [lambda: 0.4, lambda: 0.4, lambda: 0.4], [lambda: 0.4, lambda: 1.7, lambda: 0.4]][::-1],
            'pos': [[1.4, 0.5, 0.2], [1.8, -0.5, 0.2], [.6, 0.0, 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['fixed_asset_13', 'fixed_asset_14', 'movable_asset_8'][::-1],
            'size': [[lambda: 0.4, lambda: 0.6, lambda: 0.4], [lambda: 0.4, lambda: 0.4, lambda: 0.4], [lambda: 0.4, lambda: 1.7, lambda: 0.4]][::-1],
            'pos': [[1.4, -0.5, 0.2], [1.8, 0.5, 0.2], [.6, 0.0, 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
]

########
    #########
########
TASK_3 = [

    {
        'name': ['fixed_asset_15', 'fixed_asset_16', 'movable_asset_9'][::-1],
        'size': [[lambda: 0.3, lambda: 1.1, lambda: 0.4], [lambda: 0.3, lambda: 1.1, lambda: 0.4], [lambda: 0.3, lambda: 1.15, lambda: 0.4]][::-1],
        'pos': [[0.8, -0.3, 0.2], [1.6, -0.3, 0.2], [1.2, 0.1, 0.2]][::-1],
        'density': [10000, 10000, 8][::-1]
    },

    {
        'name': ['fixed_asset_17', 'fixed_asset_18', 'movable_asset_10'][::-1],
        'size': [[lambda: 0.3, lambda: 1.1, lambda: 0.4], [lambda: 0.3, lambda: 1.1, lambda: 0.4], [lambda: 0.3, lambda: 1.15, lambda: 0.4]][::-1],
        'pos': [[0.8, 0.3, 0.2], [1.6, 0.3, 0.2], [1.2, -0.1, 0.2]][::-1],
        'density': [10000, 10000, 8][::-1]
    }
]

#####     ######
    ########
TASK_4 = [
    
    {
        'name': ['fixed_asset_19', 'fixed_asset_20', 'movable_asset_11'][::-1],
        'size': [[lambda: 0.4, lambda: 0.7, lambda: 0.4], [lambda: 0.4, lambda: 0.7, lambda: 0.4], [lambda: 0.4, lambda: 0.8, lambda: 0.4]][::-1],
        'pos': [[1.6, -(0.93-0.35), 0.2], [1.2, (0.93-0.35), 0.2], [0.85, 0.0, 0.2]][::-1],
        'density': [10000, 10000, 8][::-1]
    },
]

##########
    ##########
TASK_5 = [
    
    {
        'name': ['fixed_asset_21', 'movable_asset_12'][::-1],
        'size': [[lambda: 0.3, lambda: 1.4, lambda: 0.4], [lambda: 0.3, lambda: 1.2, lambda: 0.4]][::-1],
        'pos': [[1.2, -(0.93-0.75), 0.2], [0.85, 0.1, 0.2]][::-1],
        'density': [10000, 8][::-1]
    },

    {
        'name': ['fixed_asset_22', 'movable_asset_13'][::-1],
        'size': [[lambda: 0.3, lambda: 1.4, lambda: 0.4], [lambda: 0.3, lambda: 1.2, lambda: 0.4]][::-1],
        'pos': [[1.2, (0.93-0.75), 0.2], [0.85, -0.1, 0.2]][::-1],
        'density': [10000, 8][::-1]
    }
]

# 3 assets with 2 fixed assets on either side of the movable asset
TASK_6 = [
    {
        'name': ['fixed_asset_23', 'fixed_asset_24', 'movable_asset_14'][::-1],
        'size': [[lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.42, lambda: 0.4]][::-1],
        'pos': [[1.2, -(0.93-0.35), 0.2], [1.2, (0.93-0.35), 0.2], [1.2, 0.0, 0.2]][::-1],
        'density': [10000, 10000, 8][::-1]
    },

    # {
#     'name': ['fixed_asset_25', 'fixed_asset_26', 'movable_asset_15'][::-1],
    #     'size': [[lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.42, lambda: 0.4]][::-1],
    #     'pos': [[1.2, -(0.93-0.35), 0.2], [1.2, (0.93-0.35), 0.2], [1.5, 0.0, 0.2]][::-1],
    #     'density': [10000, 10000, 8][::-1]
    # },

    {
        'name': ['fixed_asset_27', 'fixed_asset_28', 'movable_asset_16'][::-1],
        'size': [[lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.42, lambda: 0.4]][::-1],
        'pos': [[1.2, -(0.93-0.35), 0.2], [1.2, 0.14, 0.2], [1.2, 0.72, 0.2]][::-1],
        'density': [10000, 10000, 8][::-1]
    },

    {
        'name': ['fixed_asset_29', 'fixed_asset_30', 'movable_asset_17'][::-1],
        'size': [[lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.7, lambda: 0.4], [lambda: 0.3, lambda: 0.42, lambda: 0.4]][::-1],
        'pos': [[1.2, (0.93-0.35), 0.2], [1.2, -0.14, 0.2], [1.2, -0.72, 0.2]][::-1],
        'density': [10000, 10000, 8][::-1]
    }
]

# -0.58 + 0.72 + 0.35 =  
# INPLAY_ASSETS = [*TASK_ONE]
# EVAL_INPLAY_ASSETS = [*TASK_ONE]
INPLAY_ASSETS = [*TASK_ONE, *TASK_TWO]
EVAL_INPLAY_ASSETS = [*TASK_ONE, *TASK_TWO]
# INPLAY_ASSETS = [*TASK_ONE, *TASK_TWO, *TASK_THREE]
# EVAL_INPLAY_ASSETS = [*TASK_ONE, *TASK_TWO, *TASK_THREE]

# INPLAY_ASSETS = [*RANDOM_INPLAY_ASSETS[1:2]]
# EVAL_INPLAY_ASSETS = [*RANDOM_INPLAY_ASSETS[1:2]]

# INPLAY_ASSETS = [*TASK_0]
# EVAL_INPLAY_ASSETS = [*TASK_0]

# INPLAY_ASSETS = [*TASK_0[3:]]
# EVAL_INPLAY_ASSETS = [*TASK_0[3:]]

# INPLAY_ASSETS = [*TASK_REAL]
# EVAL_INPLAY_ASSETS = [*TASK_REAL]

# EVAL_INPLAY_ASSETS = [*TASK_1, *TASK_2, *TASK_4, *TASK_5, *TASK_6]
# INPLAY_ASSETS = [*TASK_1, *TASK_2, *TASK_4, *TASK_6]
# EVAL_INPLAY_ASSETS = [*TASK_0, *TASK_1, *TASK_2, *TASK_4]

# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-04_10-38-36'
# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-05_11-45-09'
# # # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-06_07-47-52'
# # # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-06_17-38-24'
# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-08_08-39-35'
# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-12_11-44-44'
# HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_07-02-12'


# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_10-05-03'
# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_11-59-48'
# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_13-20-46'
# # HOME_DIR = '/common/home/dm1487/robotics_research/legged_manipulation/gaited-walk/scene_predictor/results_2_obs_se/2023-09-15_16-42-21'
