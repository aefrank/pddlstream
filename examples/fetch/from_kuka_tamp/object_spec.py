from collections import UserDict
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
from typing_extensions import TypeAlias

from numpy import random

from iGibson.igibson.utils.assets_utils import (
    get_ig_model_path,
    get_all_object_categories, 
    get_object_models_of_category
    )

Args: TypeAlias = list
Kwargs: TypeAlias = Dict[str,Any]

Real : TypeAlias = Union[int,float]
UID : TypeAlias  = Union[str,int]

Position3D = NamedTuple("Position3D", x=Real, y=Real, z=Real)
Position2D = NamedTuple("Position2D", x=Real, y=Real)
Position : TypeAlias = Union[Position2D, Position3D]

Euler = NamedTuple("Euler", r=Real, p=Real, y=Real)
Quaternion : TypeAlias = Tuple[Real, Real, Real, Real]
Orientation2D : TypeAlias = Real
Orientation3D : TypeAlias = Union[Euler, Quaternion]
Orientation : TypeAlias = Union[Orientation2D, Orientation3D]


class URDFObjectSpec(UserDict):
    def __init__(self, category:str, name:Optional[str]=None, *, model:Optional[str]=None, **URDF_kwargs:Kwargs):
        if category not in get_all_object_categories():
            raise ValueError(f"Unable to find object category '{category}' in assets.")
        
        self.data = URDF_kwargs
        self.data["category"] = category
        self.data["name"] = name if (name is not None) else category

        if model is None: 
            model = random.choice(get_object_models_of_category(self.data["category"]))
        model_path = self.data["model_path"] if "model_path" in self.data else get_ig_model_path(self.data["category"], model)
        self.data["filename"] = os.path.join(model_path, model + ".urdf")
        
class ObjectSpec(UserDict):
    _states = ("position", "orientation")
    
    def __init__(self, position:Position=Position3D(0,0,0), orientation:Orientation=Euler(0,0,0), **URDF_kwargs):
        self.data = URDFObjectSpec(**URDF_kwargs)
        self.data.update({
            "position" : position,
            "orientation" : orientation
        })
    @property
    def urdf_data(self):
        return {k:v for k,v in self.data.items() if k not in ObjectSpec._states}
    @property
    def state_data(self):
        return {k:v for k,v in self.data.items() if k in ObjectSpec._states}
    @property
    def URDF(self):
        return URDFObjectSpec(**self.urdf_data)
    

def _as_urdf_spec(spec:Union[str,dict]) -> "URDFObjectSpec":
    # def _obj_attr_check(spec:dict) -> None:
    #     '''Ensure all necessary keys present in object dict, and no forbidden keys are present.
    #     '''
    #     necessary, forbidden = set(['category']), set(cls._states)
    #     missing = necessary - set(spec)
    #     extras = forbidden & set(spec)
    #     if len(missing)>0:  raise ValueError(f"object specification missing necessary key(s) '{missing}':\n{spec}")
    #     if len(extras)>0:   raise ValueError(f"object specification contains forbidden key(s) '{extras}':\n{spec}")        

    if isinstance(spec, dict) or isinstance(spec,UserDict):
        urdf_spec = ObjectSpec(**spec).URDF
    elif isinstance(spec,str): 
        urdf_spec = URDFObjectSpec(spec)
    else:
        raise TypeError(f"Inappropriate argument 'spec' of type {type(spec)}. Expected str or dict.")
    assert isinstance(urdf_spec, URDFObjectSpec)
    return urdf_spec