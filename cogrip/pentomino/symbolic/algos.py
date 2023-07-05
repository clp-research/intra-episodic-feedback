import random
from typing import List, Set, Union

from cogrip.pentomino.symbolic.types import PropertyNames, SymbolicPiece, SymbolicPieceGroup


class PentoIncrementalAlgorithm:

    def __init__(self, preference_order: List[PropertyNames], start_tokens: List = None):
        """
        :param property_names: the order does matter! since first props rule more likely some distractors out in the
        first iteration and later props less likely rule out remaining distractors (if there are any left)
        e.g. the first prop might already rule out everything and the algorithm stops, though others would do the same
        """
        self.preference_order = preference_order
        self.general_types = ["piece"]
        self.start_tokens = ["Take", "Select", "Get"]
        if start_tokens:
            self.start_tokens = start_tokens

    def generate(self, pcl: Union[SymbolicPieceGroup, List], selection: SymbolicPiece, is_selection_in_pieces=False,
                 return_expression=True):
        """
            pieces: a list of pieces (incl. the selection)
            selection: a selected pieces (within pieces)
        """
        if isinstance(pcl, SymbolicPieceGroup):
            distractors = set(pcl.pieces)
        else:
            distractors = set(pcl)
        if is_selection_in_pieces:
            distractors.remove(selection)
        # property-value pairs are collected here
        properties = {}
        for property_name in self.preference_order:
            property_value = selection[property_name]
            # check what objects would be eliminated using this prop-val pair
            excluded_distractors = self._exclude(property_name, property_value, distractors)
            if property_value and len(excluded_distractors) > 0:
                # save the property
                properties[property_name] = property_value
                # update the contrast set
                for o in excluded_distractors:
                    distractors.remove(o)
            # check if enough properties have been collected to rule out all distractors
            if not len(distractors):
                if return_expression:
                    return self.verbalize_properties(properties), properties, True
                return properties, True
        # there might be a case where no properties have been found at all (all pieces are the same)
        # in that case we might want to mention all properties (instead of saying nothing)
        if len(properties) == 0:
            properties = dict([(pn, selection[pn]) for pn in list(PropertyNames)])
        if return_expression:
            return self.verbalize_properties(properties, False), properties, False
        return properties, False

    def verbalize_properties(self, properties, is_discriminating=True):
        start_token = random.choice(self.start_tokens)
        shape = properties[PropertyNames.SHAPE] \
            if PropertyNames.SHAPE in properties and PropertyNames.SHAPE in self.preference_order \
            else random.choice(self.general_types)
        color = properties[PropertyNames.COLOR] \
            if PropertyNames.COLOR in properties and PropertyNames.COLOR in self.preference_order \
            else ""
        pos = f"at the {properties[PropertyNames.REL_POSITION]}" \
            if PropertyNames.REL_POSITION in properties and PropertyNames.REL_POSITION in self.preference_order \
            else ""
        ref_exp = f"{color} {shape} {pos}".strip()  # strip whitespaces if s.t. is empty
        if is_discriminating:
            return f"{start_token} the {ref_exp}"
        return f"{start_token} one of the {ref_exp}"

    def _exclude(self, property_name: PropertyNames, selection_property_value, distractors: Set[SymbolicPiece]):
        """
         * Helper function for IA.
         * Params:
         * property_name - property name
         * selection_property_value - value the target object has assigned to the given property
         * distractors - contrast set of objects to rule out by the given properties
         * Returns:
         * array of contrast objects with a different value for prop
        """
        excluded_distractors = []
        for distractor in distractors:
            if distractor[property_name] != selection_property_value:
                excluded_distractors.append(distractor)
        return excluded_distractors
