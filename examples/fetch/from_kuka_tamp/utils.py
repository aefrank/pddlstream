
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Union
import numpy as np
# from typing_extensions import TypeAlias
# import os
# from os import PathLike
import importlib
# from inspect import ismodule as is_module_object
import inspect as ins

PybulletToolsVersion = Enum('PybulletToolsVersion', ['PDDLSTREAM', 'IGIBSON'])
STREAM = PybulletToolsVersion.PDDLSTREAM
IGIBSON = PybulletToolsVersion.IGIBSON
UTILS = {
    STREAM : 'examples.pybullet.utils',
    IGIBSON : 'igibson.external'
}

def import_from(module, targets=[], package=None):
    if not targets:
        return tuple()
    def import_targets_from_object(module, *targets):
        targets = tuple(getattr(module,target) for target in targets)
        return targets if len(targets)>1 else targets[0]
    def import_module_with_targets(module_name, *targets, package=None):
        if package is not None:
            if ins.ismodule(package):
                if hasattr(package, module_name):
                    return getattr(package, module_name)
                else:
                    package = package.__name__
            assert isinstance(package,str)
            if not module_name.startswith("."):
                module_name = "." + module_name
            module_name = package + module_name
        assert isinstance(module_name,str)
        return __import__(module_name, fromlist=targets) # this "should" make sure the targets are loaded

    if ins.ismodule(module):
        try:
            return import_targets_from_object(module, *targets)
        except Exception:
            # re-import module
            module = module.__name__
    assert isinstance(module,str)
    module = import_module_with_targets(module, *targets, package=package)
    return import_targets_from_object(module, *targets)


    


def import_module(module, package=None):
    if module=='motion':
        module = 'motion.motion_planners'
    elif module=='pybullet_tools':
        module = 'pybullet_tools'

    if ins.ismodule(package):
        if hasattr(package,module):
            return getattr(package,module)
        else:
            package = package.__name__
        
    if package is not None:
        if not module.startswith("."):
            module = "." + module
        if isinstance(package, PybulletToolsVersion):
            package = UTILS[package]

    return importlib.import_module(module, package)



def elements_satisfy(iterable:Iterable, condition:Callable[[Any],bool])->bool:
    return all(condition(elem) for elem in iterable)

def is_iterable_with(
        iterable:Iterable, 
        condition:Optional[Callable[[Iterable],bool]]=lambda it: True, 
        element_condition:Optional[Callable[[Any],bool]]=None
) -> bool:
    return (
        # confirm that iterable is indeed an Iterable
        hasattr(iterable, "__iter__") and               
        # if condition was specified, confirm that iterable satisfies it
        (condition is None or condition(iterable)) and  
         # if element_condition was specified, confirm that every element in iterable satisfies it
        (element_condition is None or all(element_condition(elem) for elem in iterable))   
    )


def is_numeric(x:Any):
    try: 
        float(x)
        return True
    except Exception:
        return False
    # return isinstance(x,(int,float))

def is_numeric_vector(v:Any):
    # return hasattr(v, "__iter__") and all(is_numeric(x) for x in v)
    return is_iterable_with(v, element_condition=is_numeric)

def is_nonstring_iterable(x:Any) -> bool:
    return hasattr(x, "__iter__") and not isinstance(x, str)
    
def nonstring_iterable(v:Any) -> List:
    if is_nonstring_iterable(v): return list(v)
    elif isinstance(v,str): return [v]
    else: return [v] # non-iterable non-string


def eagermap(__func, *__iterables):
    '''Eager evaluation of mapped function on iterables.
    '''
    return [__func(*elems) for elems in zip(*__iterables)]

def get_structure_safe_map(__map):
    def maintain_type(_original, _mapped):
        return type(_original)(_mapped) if not isinstance(_original, np.ndarray) else np.array(_mapped)
    def structure_safe_map(__func, __iterable):
        return maintain_type(__iterable, __map(__func, __iterable))
    return structure_safe_map

def get_recursive_map(base_case=(lambda elem: not is_nonstring_iterable(elem)), local_map=map):
    def recursive_map(__func, __iterable):
        def _recursion(__current_node):
            return __func(__current_node) if base_case(__current_node) else local_map(_recursion, __current_node)
        return local_map(_recursion, __iterable)
    return recursive_map

def recursive_map(fn, itr, base_case=(lambda elem: not is_nonstring_iterable(elem)), lazy=False, preserve_iterable_types=True):
    _map = map if lazy else eagermap
    _map = get_structure_safe_map(_map) if preserve_iterable_types else _map
    recursive_map = get_recursive_map(base_case, _map)
    return recursive_map(fn, itr)
            

def recursive_map_advanced(
    fn, 
    itr, 
    base_case=(lambda elem: not is_nonstring_iterable(elem)), 
    *,
    nonterminal_pre_recursion_fn=None,
    nonterminal_post_recursion_fn=None,
    lazy=False, 
    preserve_iterable_types=True,
):
    
    def get_pre_postprocess_map(preprocess_fn=None, postprocess_fn=None, __map=map):
        if preprocess_fn and postprocess_fn:
            return lambda __func, __iterable: postprocess_fn(__map(__func, preprocess_fn(__iterable)))
        elif not preprocess_fn and postprocess_fn:
            return lambda __func, __iterable: postprocess_fn(__map(__func, __iterable))
        elif not postprocess_fn and preprocess_fn:
            return lambda __func, __iterable: __map(__func, preprocess_fn(__iterable))
        else:
            return __map
    
    _map = map if lazy else eagermap
    _map = get_pre_postprocess_map(nonterminal_pre_recursion_fn, nonterminal_post_recursion_fn, _map)
    _map = get_structure_safe_map(_map) if preserve_iterable_types else _map
    recursive_map = get_recursive_map(base_case, _map)
    return recursive_map(fn, itr)
    
def recursive_map_custom(
    fns,  #(fn:Callable[[iterable],iterable], condition:Callable[[iterable],bool], direction={'prefix','postfix','both',None} (optional))
    iterables, 
    lazy=False, 
):
    def vacuously_true(x):
        return any(x == [], x is None, x is True)
    
    def apply_fns(x, current_direction):
        for fn,*condition in fns:
            if len(condition)>1:
                condition, direction = condition
            else:
                condition, direction = condition, None
            
            if vacuously_true(direction) or direction == "both":
                direction = True            
            # assert direction in {"prefix", "postfix", "both"}
            # assert current_direction in {"prefix", "postfix", "both"}
            if direction:
                if vacuously_true(condition) or condition(x):
                    condition = True
                x = fn(x)
        return x
    

    
def round_numeric(x:Any, dec:int=3, **kwargs):
    '''Round numeric x or elements of x (if x is iterable) to 'dec' places if numeric. 
    Any non-numeric x or element of x is returned/included as is.
    '''
    
    def _recursion(*elems:List[Any], lazy=False) -> List[Any]:
        '''Recursively traverse x and round any numeric values found to 'dec' places. 
        '''
        array = lambda arr: np.array(arr)
        constr = lambda itr: type(itr) if not isinstance(itr, np.ndarray) else array # np.ndarray constructor edge case
        
        eagermap = lambda func, itr: constr(itr)([func(e) for e in itr]) # force eager evaluation
        _map = map if lazy else eagermap

        base_case = lambda elem: round(elem,3) if is_numeric(elem) else elem
        recursive_case = lambda elem: _map(recursive_case,elem) if is_nonstring_iterable(elem) else base_case(elem)
        # return _map(recursive_case, elems)
    
        return recursive_map(base_case, elems, lazy=False, **kwargs)

    
    y, *_ = _recursion(x) # y will be wrapped in an extra layer
    return y


    
# def is_resource(x:Any) -> bool:
#     isinstance(x,(str,ModuleType))
# Module:TypeAlias = Union[ModuleType,Resource]
# def is_module(x:Any) -> bool:
#     return is_resource(x) or is_module_object(x)

# import numpy as np





# # def import_module(*module:List[Resource], package:Optional[Package]=None):
# #     if len(module)==0:
# #         raise ValueError("Must pass at least one 'module'")
# #     return [importlib.import_module(mod, package) for mod in module]

# def import_member(member:str, module:Module):
#     if is_resource(module):
#         module = importlib.import_module(module)
#     assert is_module_object(module)
#     return getattr(module, member)

# def import_members(module:Module, *members:List[str]):
#     if is_resource(module):
#         module = importlib.import_module(module)
#     assert is_module_object(module)
#     if members:
#         return getattr(module, [mem for mem in members])
#     else: # If empty list, import all members
#         return ins.getmembers(module)
    
# def get_from(source, *targets:List[str], parent=None):
#     '''Note: will treat string 'source' as the identifier of the source object, not the actual object to query.
#     '''
#     def __member_from_module_name(__module:str, *__member:str):
#         return __import__(__module, fromlist=__member)
#     def __module_from_none(__module:str):
#         return importlib.import_module(__module)
#     def __module_from_package_name(__package:str, __module:str):
#         return importlib.import_module(__module, __package)
    
#     def __from_module_name(__module:str, *__targets:List[str], __package:Optional[str]=None):
#         if not targets:
#             raise NotImplementedError
#         else:
#             # module = importlib.import_module(__module, __package)
#             # [getattr(module, t) for t in targets]
#             module = importlib.util.resolve_name(__module, __package)
#             return __import__(module, fromlist=targets)
    
#     if isinstance(source,str): 
#         if parent is None:
#             # source must be the name/path of a module/pkg
#             source = importlib.import_module(source)
#         else:
#             if isinstance(parent, str):
#                 # parent must be the name/path of a module/pkg
#                 parent = importlib.import_module(parent)
#             # parent is an object
#             source = getattr(parent, source)
#             # now source is an object
#             return get_from(source, *targets)
#     else:
#         if len(targets)==1: # 'source' is an object itself as opposed to a string describing the source
#             return getattr(source, targets[0])
#         else:
#             return [getattr(source, target) for target in targets]

    
    





# def _import_from(module:Module, *members:List[str], package:Optional[Package]=None): 
#     '''
#     from module import member           =>          member = import_from("module", "member")   
#     import module;                                    OR          import module; member = import_from(module, "member")

#     from module import member as alias  =>          alias = import_from("module", "member") 
#                                         OR          import module; alias = import_from(module, "member") 

#     from module import member1, member2 =>          member1, member2 = import_from("module", "member1", "member2")
    
#     import package.module as module                 => import_from("module", package=package) 
#     import package.module                           => package = import_from(None, package=package)



#         Before execution:
#             variable package may or may not be defined in global scope
        
#         After execution:
#             variable package is unchanged (will not be defined if was not beforehand)
#             variable <module:ModuleType> defined in global scope, with attributes module.__name__="module" and module.__package__=package.__name__
    
#     module =       
#     import package; from package import module      import package; import_from("module", package=package)
                                        

    
#     '''
#     # Type checking
#     ## Several checks happen more than once -- I went more for readability than efficiency
#     # if package is None:
#     #     if is_resource(module):
#     #         module = importlib.import_module(module)

#     # assert package is None or is_package(package)
#     # # Check package type, which affects If package is defined, that implies we need to import module from package, so module must be a Resource, not a ModuleType object
#     # assert package is None or (is_package(package) and not is_module_object(module)) 
        

# # def _import(module:Module, package:Optional[Package]=None, members:Iterable[str]=[]):
# #     '''Dynamically import module, or 
# #     '''
# #     assert package is None or is_package(package)
# #     assert is_module(module)

# #     if is_resource(module):
# #         module = importlib.import_module(module, package)
# #     if not members:
# #         return module
# #     else:
# #         return [getattr(module, member) for member in members]
#         # return __import__(module, fromlist=members)


# # def import_from(attr:str, module:Union[Resource,ModuleType], package:Optional[Union[Resource,Package]]=None):
    
# #     return getattr(import_module(module, package), attr)
