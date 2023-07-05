from typing import Dict, List, Union

from cogrip.core import Jsonable
from cogrip.core.obj import Obj


class Gripper(Obj):
    def __init__(
            self, id_n, x, y, gripped_obj: Obj = None, color="blue"):
        super().__init__(
            id_n, "gripper", x, y, [[1]],
            rotation=0, mirrored=False, color=color, gripped=False
        )
        self.gripped_obj = gripped_obj

    @classmethod
    def from_dict(cls, source_dict, type_config=None):
        """
        Construct a new Gripper instance from a dictionary, e.g., parsed json.
        @param source_dict  dict containing object attributes, keys "x"
                            and "y" are mandatory
        @param type_config  dict mapping type names to block matrices
        @return new Gripper instance with the given attributes
        """
        mandatory_key = {"x", "y"}
        if any(source_dict.get(key) is None for key in mandatory_key):
            raise KeyError(
                f"Gripper construction failed, key {mandatory_key} missing"
            )

        new_gripper = cls(
            source_dict["id_n"],
            float(source_dict["x"]),
            float(source_dict["y"])
        )

        # process optional info
        if "gripped" in source_dict:
            # cast object name to str too
            gripped_id = str(source_dict["gripped"])
            new_gripper.gripped = gripped_id

        if "width" in source_dict:
            new_gripper.width = source_dict["width"]
        if "height" in source_dict:
            new_gripper.height = source_dict["height"]
        if "color" in source_dict:
            new_gripper.color = source_dict["color"]
        return new_gripper

    def to_dict(self):
        """
        Constructs a JSON-friendly dictionary representation of this instance.
        @return dictionary containing all important properties
        """
        return {
            "id_n": self.id_n,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "gripped_obj": None
        }

    def to_json(self):
        data = self.to_dict()
        if self.gripped_obj:
            data["gripped_obj"] = self.gripped_obj.to_json()
        return data


class Grippers(Jsonable):
    """ A collection of gripper """

    def __init__(self, grippers: List[Gripper] = None):
        self.grippers_by_id: Dict[int, Gripper] = dict([(obj.id_n, obj) for obj in grippers]) if grippers else {}

    def __getitem__(self, item):
        return self.get_gripper_by_id(item)

    def __len__(self):
        return len(self.grippers_by_id)

    def __iter__(self):
        return self.grippers_by_id.values().__iter__()

    def add(self, gr: Gripper):
        self.grippers_by_id[gr.id_n] = gr
        return gr

    def remove(self, gr: Union[Gripper, int]):
        idx = gr
        if isinstance(gr, Gripper):
            idx = gr.id_n
        del self.grippers_by_id[idx]
        return gr

    def get_gripper_by_id(self, gr_id):
        if gr_id in self.grippers_by_id:
            return self.grippers_by_id[gr_id]
        return None

    def get_coords_for(self, gr_id):
        """
        @param gr_id 	gripper id
        """
        if gr_id in self.grippers_by_id:
            gripper = self.grippers_by_id[gr_id]
            return [gripper.x, gripper.y]
        return []

    def get_gripped_obj_for(self, gr_id):
        """
        @param gr_id 	gripper id
        @return None or the id of the gripped object
        """
        if gr_id in self.grippers_by_id:
            return self.grippers_by_id[gr_id].gripped_obj
        else:
            return None

    def to_json(self):
        """
        In contrast to get_obj_dict, each gripper dict has
        the entry "gripped", which itself is None or a
        dictionary mapping the gripped object to an object dictionary.
        @return Dictionary mapping gripper ids to gripper dictionaries.
        """
        data = dict()
        for gr_id, gripper in self.grippers_by_id.items():
            data[gr_id] = gripper.to_dict()
        return data
