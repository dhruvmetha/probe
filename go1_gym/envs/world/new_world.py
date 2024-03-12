from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from abc import ABC, abstractmethod

class World(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset_idx(self, env_ids):
        raise NotImplementedError
    
    @abstractmethod
    def pre_create_actor(self):
        pass

    @abstractmethod
    def create_actor(self, env_id, env_handle, env_origin):
        raise NotImplementedError

    @abstractmethod
    def post_create_actor(self, env_handles):
        raise NotImplementedError
    
    
class WorldAsset(World):
    def __init__(self, gym, sim, num_envs, env_origins, n_boxes, 
                 device, asset_list, train_ratio=0.95):
        
        self.gym = gym
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self.n_boxes = n_boxes
        self.env_origins = env_origins
        self.train_ratio = train_ratio
        self.asset_list = asset_list

        self._init_buffers()

    def _init_buffers(self):

        self.box_loc = torch.zeros((self.num_envs, self.n_boxes, 2), dtype=torch.float32, device=self.device)
        self.box_size = torch.zeros((self.num_envs, self.n_boxes, 2), dtype=torch.float32, device=self.device)

    def create_assets(self, env_id):
        
        assets_to_create = self.asset_list[env_id]
        assets_container = []

        for asset in enumerate(assets_to_create):
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = False

            size = asset['size']
            mass = np.random.uniform(0.5, 1.5)
            volume = np.prod(size)
            density = mass/volume
            asset_options.density = density
            box_friction = np.random.uniform(0.5, 1.2)
            box_restitution = np.random.uniform(0.0, 0.1)

            box_asset = self.gym.create_box(self.sim, *size, asset_options)
            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(box_asset)

            for s in range(len(rigid_shape_props_asset)):
                rigid_shape_props_asset[s].friction = box_friction
                rigid_shape_props_asset[s].restitution = box_restitution
            
            self.gym.set_asset_rigid_shape_properties(box_asset, rigid_shape_props_asset)

            self.box_loc[env_id]

            



    

    
