from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch
import time
import random

from go1_gym.envs.world.world_config import *


# np.random.seed(42)
# torch.manual_seed(42)

class AssetDef:
    def __init__(self, asset, name, base_position) -> None:
        self.asset = asset
        self.name = name
        self.base_position = base_position

class WorldSetup:
    def __init__(self):
        pass
    def define_world(self):
        pass
    def reset_world(self):
        pass

class WorldAsset():
    def __init__(self, gym, sim, num_envs, env_origins, device, train_ratio=0.95) -> None:
        """
        Creates the world environment objects like obstacles, walls, etc.
        """
        self.envs = None # gets initialized in post_create_world
        self.gym = gym
        self.sim = sim
        self.num_envs = num_envs # gets initialized in post_create_world
        self.num_train_envs = max(1, int(self.num_envs*train_ratio)) # gets initialized in post_create_world
        self.num_eval_envs = self.num_envs - self.num_train_envs # gets initialized in post_create_world
        self.device = device
        self.env_origins = env_origins

        self.custom_box = world_cfg.CUSTOM_BLOCK


        # initialize buffers and variables needed for the world actors
        self.handles = {}
        self.env_assets_map = {}
        self.env_actor_indices_map = {}
        self.all_actor_base_postions = {}
        self.variables = {}

        self.contact_memory_time = 20
        self.reset_timer_count = 20

        self.inplay_assets = INPLAY_ASSETS
        self.eval_inplay_assets =  EVAL_INPLAY_ASSETS # INPLAY_ASSETS

        self.inplay = {}
        self.world_types_success = {}
        self.train_eval_assets = {}

        self.base_assets = []
        self.asset_counts = 0
        self.asset_idx_map = {}
        self.idx_asset_map = {}
        self.env_assetname_map = {i: {} for i in range(self.num_envs)}
        self.env_assetname_bool_map = {i: {} for i in range(self.num_envs)}

        self.world_types = 0
        self.eval_world_types = 0
        
        idx_ctr = 0
        for i in self.inplay_assets:
            self.world_types_success[self.world_types] = 0
            self.world_types += 1
            self.asset_counts += len(i['name'])
            for j in i['name']:
                self.train_eval_assets[j] = False
                self.asset_idx_map[j] = idx_ctr
                self.idx_asset_map[idx_ctr] = j
                idx_ctr += 1


        for i in self.eval_inplay_assets:
            self.asset_counts += len(i['name'])
            self.eval_world_types += 1
            for j in i['name']:
                name_eval = j + '_eval'
                self.train_eval_assets[name_eval] = True
                self.asset_idx_map[name_eval] = idx_ctr
                self.idx_asset_map[idx_ctr] = name_eval
                idx_ctr += 1

        self.block_size = torch.zeros((self.num_envs, self.asset_counts, 2), device=self.device, dtype=torch.float, requires_grad=False)
        self.block_weight = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.float, requires_grad=False)

        self.block_contact_buf = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.block_contact_ctr = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.fixed_block_contact_buf = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.fixed_block_contact_ctr = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.inplay_env_world = torch.zeros((self.num_envs, self.world_types), device=self.device, dtype=torch.bool, requires_grad=False)
        self.env_world_success = torch.zeros(self.world_types, device=self.device, dtype=torch.int, requires_grad=False)
        self.env_world_counts = torch.zeros(self.world_types, device=self.device, dtype=torch.int, requires_grad=False)

        self.eval_worlds = torch.zeros(self.num_eval_envs, device=self.device, dtype=torch.int, requires_grad=False)
        self.total_eval_worlds = (self.num_eval_envs//self.eval_world_types) * self.eval_world_types
        
        if self.total_eval_worlds == 0:
             self.eval_worlds[:] = torch.randint(0, self.eval_world_types, (self.num_eval_envs,))
        else:
            self.eval_worlds[:self.total_eval_worlds] = torch.arange(0, self.eval_world_types).view(-1, 1).repeat(1, self.num_eval_envs//self.eval_world_types).view(-1)
            self.eval_worlds[self.total_eval_worlds:] = torch.randint(0, self.eval_world_types, (self.num_eval_envs - self.total_eval_worlds,))
            
        self.world_sampling_dist = torch.zeros(self.world_types, device=self.device, dtype=torch.float, requires_grad=False) + 1/self.world_types

        self.base_assets = []
        if True:
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = True

            gym_assets = [ \
                self.gym.create_box(self.sim, 9.0, .1, 1., asset_options), \
                self.gym.create_box(self.sim, 9.0, .1, 1., asset_options), \
                self.gym.create_box(self.sim, 0.1, 3.05, 1., asset_options), \
                self.gym.create_box(self.sim, 0.1, 3.05, 1., asset_options), \
                    ]
            asset_names = ['wall_left' , 'wall_right', 'wall_back', 'wall_front']

            # all base positions
            asset_pos = [[1., -1.05, .5],  [1., 1.05, .5], [-0.75, 0., .5], [4.0, 0., 0.5]]

            self.base_assets = [AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)]
        
        self.per_rect = 7 # PER_RECT

        self.env_asset_ctr = torch.zeros(self.num_envs, self.asset_counts, self.per_rect, dtype=torch.long, device=self.device)
        self.env_asset_ctr[:, :, :] = torch.arange(0, self.per_rect, dtype=torch.long, device=self.device)
        self.env_asset_bool = torch.zeros(self.num_envs, self.asset_counts, 1, dtype=torch.bool, device=self.device)
        self.all_train_ids = torch.arange(0, self.num_train_envs, 1, dtype=torch.long, device=self.device).view(-1, 1)
        self.all_eval_ids = torch.arange(self.num_train_envs, self.num_envs, 1, dtype=torch.long, device=self.device).view(-1, 1)

        self.base_positions = torch.zeros(self.num_envs, 3, device=self.device)

        self.reset_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) + self.reset_timer_count
        self.full_info_bool = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.full_info_prob = 0.0
        self.full_info_decay_rate = 0.95
        self.last_obs = None

    def define_world(self, env_id):
        """
        define the world configuration and it's assets
        """
        assets_container = []
        assets_inplay = self.inplay_assets if env_id < self.num_train_envs else self.eval_inplay_assets

        for i in assets_inplay:
            gym_assets = []
            asset_names = []
            asset_pos = []

            for idx, name in enumerate(i['name']):
                asset_options = gymapi.AssetOptions()
                asset_options.disable_gravity = False
                asset_options.fix_base_link = False

                sizes = [j() for j in i['size'][idx]]
                volume = np.prod(sizes)
                mass = np.random.uniform(0.9, 1.0)
                density = mass/volume

                if 'fixed' in name:
                    asset_options.density = 1000 # i['density'][idx]
                else:
                    asset_options.density = density

                gym_assets.append(self.gym.create_box(self.sim, *sizes, asset_options))
                final_name = name if env_id < self.num_train_envs else name + '_eval'
                
                self.block_size[env_id, self.asset_idx_map[final_name], :] = torch.tensor(sizes[:2])
                self.block_weight[env_id, self.asset_idx_map[final_name], :] = torch.tensor(mass)
                asset_names.append(final_name)
                
                asset_pos.append(i['pos'][idx])
            assets_container.append([AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)])

        return assets_container

    def add_variables(self, **kwargs):
        print('here', kwargs)
        self.variables = kwargs

    def decay_full_info_prob(self):
        self.full_info_prob *= self.full_info_decay_rate


    def pre_create_actor(self):
        pass

    def create_actor(self, env_id, env_handle, env_origin):
        """
        environment setup, all the actors of the world get created here
        """
        if env_id not in self.handles:
            self.handles[env_id] = []

        for asset in self.base_assets:
            pose = gymapi.Transform()
            pos = env_origin.clone(); pos[0] += asset.base_position[0]; pos[1] += asset.base_position[1]; pos[2] += asset.base_position[2]
            pose.p = gymapi.Vec3(*pos)
            ah = self.gym.create_actor(env_handle, asset.asset, pose, asset.name, env_id, 0, 0)
            
        assets_container = self.define_world(env_id)
        self.env_assets_map[env_id] = assets_container

        for asset_container in assets_container:
            for asset in asset_container:
                pose = gymapi.Transform()
                pos = env_origin.clone(); pos[0] += asset.base_position[0]; pos[1] += asset.base_position[1]; pos[2] += asset.base_position[2]
                pose.p = gymapi.Vec3(*pos)
                ah = self.gym.create_actor(env_handle, asset.asset, pose, asset.name, env_id, 0, 0)
                if 'mov' in asset.name:
                    self.gym.set_rigid_body_color(env_handle, ah, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1.0, 0.75, 0.))
                else:
                    self.gym.set_rigid_body_color(env_handle, ah, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.823, 0.02, 0.18))
                    
                self.handles[env_id].append(ah)

        self.asset_root_ids = {}
        self.asset_contact_ids = {}
        
    def post_create_actor(self, envs):
        """
        setup indices for resets of only these world actors
        """
        self.envs = envs

        for name in self.asset_idx_map.keys():
            # print(name)
            if not self.train_eval_assets[name]:
                self.asset_root_ids[name] = torch.tensor([self.gym.find_actor_index(self.envs[i], name, gymapi.DOMAIN_SIM) for i in range(self.num_train_envs)], dtype=torch.long, device=self.device)
                block_actor_handles = [self.gym.find_actor_handle(self.envs[i], name) for i in range(self.num_train_envs)]            
                self.asset_contact_ids[name] = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs[:self.num_train_envs], block_actor_handles)], dtype=torch.long, device=self.device)
            else:
                self.asset_root_ids[name] = torch.tensor([self.gym.find_actor_index(self.envs[i], name, gymapi.DOMAIN_SIM) for i in range(self.num_train_envs, self.num_envs)], dtype=torch.long, device=self.device)
                block_actor_handles = [self.gym.find_actor_handle(self.envs[i], name) for i in range(self.num_train_envs, self.num_envs)]            
                self.asset_contact_ids[name] = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs[self.num_train_envs:self.num_envs], block_actor_handles)], dtype=torch.long, device=self.device)

    def init_buffers(self, actor_root_state, dof_state_tensor, net_contact_forces, rigid_body_state):
        """
        world buffers for actor root states, rb, contacts
        """
        self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.all_dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.all_rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.all_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        

    def _get_random_idx(self, env_id):
        if False and (env_id < self.num_train_envs):
            return torch.multinomial(self.world_sampling_dist, 1)
        else:
            assets_container = self.env_assets_map[env_id]
            if env_id >= self.num_train_envs:
                if self.total_eval_worlds == 0:
                    return torch.randint(0, len(assets_container), (1,))
                else:
                    if env_id > (self.num_train_envs + self.total_eval_worlds-1):
                        return torch.randint(0, len(assets_container), (1,))
                    else:
                        return self.eval_worlds[env_id - self.num_train_envs]
            return torch.randint(0, len(assets_container), (1,))

    
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_block_obs()
        
    def reset_idx(self, env_ids):
        """
        reset the world actors in the environment
        """
        # TODO: precompute these lists for more efficiency in post_create_world

        actor_indices = []
        base_positions = []
        env_origins = []
        env_asset_bool = []
        self.env_asset_bool[env_ids, :, :] = False
        if self.last_obs is not None:
            self.last_obs[env_ids, :,] = 0.

        for env_id in env_ids:
            env_id = env_id.item()
            if env_id < self.num_train_envs:
                self.full_info_bool[env_id] = np.random.uniform() < self.full_info_prob
            env_handle = self.envs[env_id]
            assets_container = self.env_assets_map[env_id]

            random_idx = self._get_random_idx(env_id)
            self.inplay_env_world[env_id, :]  = False
            if env_id < self.num_train_envs:
                self.inplay_env_world[env_id, random_idx] = True
                self.inplay[env_id] = random_idx
            in_indices = []
            assets_marked = []

            current_ctr = 1

            for idx, asset_container in enumerate(assets_container):
                mv_size = None
                movable_bp = None
                movable_asset_name = None 
                fixed_bp = None
                fb_three_bool = [False, False, False]
                bb_three_bool = [False, False, False]

                for asset in asset_container:
                    actor_indices.append(self.gym.find_actor_index(env_handle, asset.name, gymapi.DOMAIN_SIM))
                    env_origins.append(self.env_origins[env_id])
                    assets_marked.append(asset.name)
                    # self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] = torch.arange(0, self.per_rect, dtype=torch.long, device=self.device)
                    
                    if idx == random_idx:

                        if 'movable' in asset.name:
                            self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] = torch.arange(0, self.per_rect, dtype=torch.long, device=self.device)
                        elif 'fixed' in asset.name:
                            self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] = torch.arange(self.per_rect*current_ctr, int(self.per_rect*(current_ctr+1)), dtype=torch.long, device=self.device)

                            current_ctr += 1
                        # print(asset.name, self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :])

                        # self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] += self.per_rect*current_ctr
                        
                        if asset.name.startswith('fb_three_mov'):
                            if fixed_bp is not None:
                                raise "should do movable first"
                            mv_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            mv_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            mv_y_range = (0.975 - mv_size_y/2)

                            movable_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-mv_y_range, mv_y_range]), 2), 0.2]

                            movable_asset_name = asset.name
                            mv_size = (mv_size_x, mv_size_y)

                            base_positions.append(movable_bp)
                            self.base_positions[env_id, 0] = movable_bp[0]
                            self.base_positions[env_id, 1] = movable_bp[1]
                            fb_three_bool[0] = True
                            # print('assign', self.base_positions[env_id, :], movable_bp)

                        elif asset.name.startswith('fb_three_fix'):
                            
                            mv_x, mv_y, _ = movable_bp
                            mv_size_x, mv_size_y = mv_size
                            fx_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            fx_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            if not fb_three_bool[1]:
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x-mv_size_x/2-fx_size_x/2-np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                fb_three_bool[1] = True


                            elif fb_three_bool[1] and not fb_three_bool[2]:
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x+mv_size_x/2+fx_size_x/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                fb_three_bool[2] = True

                            base_positions.append(fixed_bp)

                        elif asset.name.startswith('bb_three_mov'):
                            if fixed_bp is not None:
                                raise "should do movable first"
                            mv_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            mv_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            mv_y_range = (0.975 - mv_size_y/2)

                            movable_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-mv_y_range, mv_y_range]), 2), 0.2]

                            movable_asset_name = asset.name
                            mv_size = (mv_size_x, mv_size_y)
                            self.base_positions[env_id, 0] = movable_bp[0]
                            self.base_positions[env_id, 1] = movable_bp[1]

                            # print('assign', self.base_positions[env_id, :], movable_bp)

                            base_positions.append(movable_bp)
                            bb_three_bool[0] = True
                        
                        elif asset.name.startswith('bb_three_fix'):
                            
                            mv_x, mv_y, _ = movable_bp
                            mv_size_x, mv_size_y = mv_size
                            fx_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            fx_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            if not bb_three_bool[1]:
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x+mv_size_x/2+fx_size_x/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                bb_three_bool[1] = True

                            elif bb_three_bool[1] and not bb_three_bool[2]:
                                bb_offset = fixed_bp[0]
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [bb_offset+0.4+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                bb_three_bool[2] = True

                            base_positions.append(fixed_bp)

                        elif asset.name == world_cfg.movable_block.name or asset.name.startswith('movable_block'):
                            if fixed_bp is not None:
                                raise "should do movable first"
                                exit()

                            mv_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()
                            mv_y_range = (0.975 - mv_size_y/2)
                            
                            movable_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-mv_y_range, mv_y_range]), 2), 0.2]
                            movable_asset_name = asset.name

                            self.base_positions[env_id, 0] = movable_bp[0]
                            self.base_positions[env_id, 1] = movable_bp[1]
                            base_positions.append(movable_bp)
                            # print('assign', self.base_positions[env_id, :], movable_bp)

                        elif asset.name == world_cfg.fixed_block.name or asset.name.startswith('fixed_block'):
                            fx_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()
                            fx_y_range = (0.975 - fx_size_y/2)
                            if movable_bp is not None:
                                mv_x, mv_y, _ = movable_bp
                                mv_size_x = self.block_size[env_id, self.asset_idx_map[movable_asset_name], 0]
                                fx_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()

                                # if np.random.uniform() < 0.5 and fx_size_y < 0.8:
                                #     fixed_bp = [mv_x-mv_size_x/2-fx_size_x/2-np.random.uniform(0, 0.2), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                # else:
                                fixed_bp = [mv_x+mv_size_x/2+fx_size_x/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                            else:
                                fixed_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                            base_positions.append(fixed_bp)

                        else:
                            base_positions.append(asset.base_position)
                    
                        # current_ctr += 1

                        # if 'movable' in asset.name:
                        #     self.base_positions[env_id, 0] = movable_bp[0]
                        #     self.base_positions[env_id, 1] = movable_bp[1]

                    else:
                        base_positions.append([bp - 100000.0 - np.random.uniform(0, 1000) for bp in asset.base_position])
                    
        actor_indices = torch.tensor(actor_indices, dtype=torch.long, device='cuda:0')
        base_positions = torch.tensor(base_positions, dtype=torch.float32, device='cuda:0')

        
        env_origins = torch.vstack(env_origins)
        self.all_root_states[actor_indices, :3] = base_positions + env_origins
        self.all_root_states[actor_indices, 3:] = 0.
        self.all_root_states[actor_indices, 6] = 1. 
        self.block_contact_buf[env_ids, :, :] = False
        self.block_contact_ctr[env_ids, :, :] = self.contact_memory_time + 1
        self.reset_timer[env_ids] = self.reset_timer_count

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_root_states), gymtorch.unwrap_tensor(actor_indices.to(dtype=torch.int32)), len(actor_indices))

    def get_block_obs(self):

        ids = []
        block_actor_handles = []
        ids_contact = []
        all_obs = torch.zeros((self.num_envs, 7*3), dtype=torch.float, device=self.device)
        full_seen_obs = torch.zeros_like(all_obs)
        if self.last_obs is None:
            self.last_obs = torch.zeros_like(all_obs) 
        # self.env_asset_ctr[:, :] = torch.arange(0, 10, dtype=torch.long, device=self.device)
        asset_keys = list(self.asset_idx_map.keys())
        # random.shuffle(asset_keys)
        for _, name in enumerate(asset_keys):
            
            curr_env_ids = self.all_train_ids.view(-1)
            if self.train_eval_assets[name]:
                curr_env_ids = self.all_eval_ids.view(-1)
            if len(curr_env_ids) == 0:
                continue 
            
            ids = self.asset_root_ids[name]
            ids_contact = self.asset_contact_ids[name]

            movable_indicator = 0
            if 'mov' in name:
                movable_indicator = 1

            self.env_asset_bool[curr_env_ids, self.asset_idx_map[name], :] = ~(self.all_root_states[ids, 0] < -100).view(-1, 1)

            rot = self.all_root_states[ids, 3:7]
            angle = torch.atan2(2.0*(rot[:, 0]*rot[:, 1] + rot[:, 3]*rot[:, 2]), 1. - 2.*(rot[:, 1]*rot[:, 1] + rot[:, 2]*rot[:, 2]))

            # print(angle.shape, (self.all_root_states[ids, :2] - self.env_origins[curr_env_ids, :2]).shape)

            all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] += (torch.cat([torch.tensor([1, movable_indicator], device=self.device).repeat(len(curr_env_ids), 1), (self.all_root_states[ids, :2] - self.env_origins[curr_env_ids, :2]).clone(), angle.view(-1, 1), self.block_size[curr_env_ids, self.asset_idx_map[name], :].clone()], dim=-1) * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]])

            full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] += (torch.cat([torch.tensor([1, movable_indicator], device=self.device).repeat(len(curr_env_ids), 1), (self.all_root_states[ids, :2] - self.env_origins[curr_env_ids, :2]).clone(), angle.view(-1, 1), self.block_size[curr_env_ids, self.asset_idx_map[name], :].clone()], dim=-1) * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]])


            # contact_forces = self.all_contact_forces[ids_contact]
            block_contact_buf = (torch.linalg.norm(self.all_contact_forces[ids_contact, :2], dim=-1) > 1.).view(-1, 1) 
            block_contact_buf_1 = (torch.linalg.norm(self.all_contact_forces[ids_contact, :2], dim=-1) > 1.).view(-1, 1) 
            
            
            moved_buf = ((torch.linalg.norm(full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:5] - self.last_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:5], dim=-1) > 0.01).view(-1, 1)) * (((torch.sum((self.last_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] > 0) * 1.0, dim=-1)) > 0)).view(-1, 1)
            
            block_contact_buf = block_contact_buf | moved_buf

            if movable_indicator:
                
                moved_from_base = (torch.linalg.norm(full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:4] - self.base_positions[curr_env_ids, :2], dim=-1) > 0.01).view(-1, 1) # * (self.reset_timer[curr_env_ids] == 0).view(-1, 1)
                block_contact_buf = block_contact_buf | moved_from_base

            # block_contact_buf *= (self.reset_timer[curr_env_ids] == 0).view(-1, 1)
            
            one_touch_bcb = block_contact_buf | (self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :] & True)
            
            self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :] = (~one_touch_bcb) * self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :]
            
            self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :] = self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :] < self.contact_memory_time
            
            self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :] += (1*self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :])

            relevant_ids = self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]].view(-1).nonzero()
            
            
            if relevant_ids.size(0) > 0:
                all_obs[curr_env_ids[relevant_ids], self.env_asset_ctr[curr_env_ids[relevant_ids], self.asset_idx_map[name]]] *= (self.block_contact_buf[curr_env_ids[relevant_ids].view(-1), self.asset_idx_map[name]] | self.full_info_bool[curr_env_ids[relevant_ids].view(-1)].view(-1, 1)) | self.variables.get('full_info', False)
                
                # if 'eval' in name:
                #     print(name, 'before', relevant_ids, block_contact_buf_1, (self.block_contact_buf[curr_env_ids[relevant_ids].view(-1), self.asset_idx_map[name]]), all_obs[curr_env_ids[relevant_ids], self.env_asset_ctr[curr_env_ids[relevant_ids], self.asset_idx_map[name]]])

                all_obs[curr_env_ids[relevant_ids], self.env_asset_ctr[curr_env_ids[relevant_ids], self.asset_idx_map[name]][:, :, 0]] *= (block_contact_buf_1[relevant_ids.view(-1)] | self.full_info_bool[curr_env_ids[relevant_ids].view(-1)].view(-1, 1)) | self.variables.get('full_info', False)

                # if 'eval' in name:
                #     print(name, 'after', relevant_ids, all_obs[curr_env_ids[relevant_ids], self.env_asset_ctr[curr_env_ids[relevant_ids], self.asset_idx_map[name]]])

        if not self.variables.get('full_info', False):
            all_obs *= (self.reset_timer == 0).view(-1, 1)
            self.reset_timer[:] -= (self.reset_timer[:] > 0).int()

        self.last_obs[:, :] = full_seen_obs[:, :]

        # print(all_obs[0])
        # print('obs', full_seen_obs[1, :4], all_obs[1, :4])
        return all_obs, full_seen_obs