__copyright__ = "Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved."
__license__ = """
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
import warnings

import numpy as np
import colorsys
import random
import omni.graph.core as og
import omni.replicator.core as rep
from omni.replicator.core import utils


class OgnSampleDefectColorInternalState:
    def __init__(self):
        self.rng = rep.rng.ReplicatorRNG()


class OgnSampleDefectColor:
    @staticmethod
    def internal_state():
        return OgnSampleDefectColorInternalState()

    @staticmethod
    def release(node):
        rep.rng.release(node.get_prim_path())

    @staticmethod
    def compute(db) -> bool:
        state = db.shared_state

        target = db.inputs.target
        num_samples = db.inputs.numSamples

        if len(target) != 3:
            return False
        
        for val in target:
            if val < 0 or val > 1:
                warnings.warn("Expected inputs:target to have values between 0 and 1")
                return False

        # validate input
        try:
            if num_samples < 1:
                warnings.warn(f"Expected inputs:num_samples to be greater than 0 but instead received {num_samples}")
                return False

            sample_size = len(target)

        except Exception as error:
            db.log_error(f"SampleUniform Error: {error}")
            return False

        is_seed_valid = db.inputs.seed is not None
        is_seed_changed = state.rng is None or db.inputs.seed != state.rng.seed
        if is_seed_valid and is_seed_changed:
            node_id = db.inputs.nodeId if db.node.get_attribute_exists("inputs:nodeId") else 0
            state.rng.initialize(db.inputs.seed, db.node, node_id)

        
        # determine ellipsoid equation
        h_upper_range = h_lower_range = .042
        s_upper_range = .2
        s_lower_range = .3
        v_upper_range = .4
        v_lower_range = .3

        h_lower = max(0, target[0] - h_lower_range)
        s_lower = max(0, target[1] - s_lower_range)
        v_lower = max(0, target[2] - v_lower_range)

        h_range = min(1, target[0] + h_upper_range) - h_lower
        s_range = min(1, target[1] + s_upper_range) - s_lower
        v_range = min(1, target[2] + v_upper_range) - v_lower

        center_h = h_lower + h_range / 2
        center_s = s_lower + s_range / 2
        center_v = v_lower + v_range / 2

        final_samples_list = []
        

        for _ in range(num_samples):
            while True: 
                h, s, v = random.random(), random.random(), random.random()

                #tup = state.rng.generator.uniform((0, 0, 0), (1, 1, 1), size=(num_samples, sample_size))
                

                # check if hsv is within an elipse determined by the target value
                if ((h - center_h)**2 / (h_range/2)**2) + ((s - center_s)**2 / (s_range/2)**2) + ((v - center_v)**2 / (v_range/2)**2) >= 1:
                    break

            final_samples_list.append(colorsys.hsv_to_rgb(h, s, v))

        samples = np.array(final_samples_list, dtype=np.float32)

        # Repeat along dimension if required
        tuple_count = db.outputs.samples.type.tuple_count
        if tuple_count > samples.shape[1]:
            samples = samples.repeat(tuple_count, 1)

        db.outputs.samples = samples
        db.outputs.numSamples = db.inputs.numSamples
        return True

    @staticmethod
    def initialize(graph_context, node):
        connected_function_callback = OgnSampleDefectColor.on_connected_callback
        node.register_on_connected_callback(connected_function_callback)

        function_callback = OgnSampleDefectColor.on_value_changed_callback
        node.get_attribute("inputs:outputType").register_value_changed_callback(function_callback)

        # Have to resolve the output type here because save and reload only calls initialize
        output_type = node.get_attribute("inputs:outputType").get()
        output_attr = node.get_attribute("outputs:samples")

        if output_type != "":
            output_attr.set_resolved_type(og.Controller.attribute_type(output_type))

    @staticmethod
    def on_connected_callback(upstream_attr, downstream_attr):
        if upstream_attr.get_name() != "outputs:samples":
            return

        node = upstream_attr.get_node()
        downstream_node = downstream_attr.get_node()

        if downstream_node.get_attribute_exists("inputs:attributeType"):
            downstream_attr_type = downstream_node.get_attribute("inputs:attributeType").get()
        else:
            downstream_attr_type = None

        output_type_attr = node.get_attribute("inputs:outputType")
        # Set the resolved type to be the downstream if it is resolved
        if upstream_attr.get_resolved_type().base_type == og.BaseDataType.UNKNOWN:
            downstream_resolved_type = downstream_attr.get_resolved_type()
            if downstream_resolved_type.base_type != og.BaseDataType.UNKNOWN:
                specified_type = downstream_resolved_type
            elif downstream_attr_type:
                downstream_attr_type = downstream_node.get_attribute("inputs:attributeType").get()
                specified_type = og.Controller.attribute_type(f"{downstream_attr_type}[]")
            else:
                specified_type = node.get_attribute("inputs:lower").get_resolved_type()
            # Get rid of role_name
            specified_type = og.Type(specified_type.base_type, specified_type.tuple_count, specified_type.array_depth)
            og.AttributeValueHelper(output_type_attr).set(str(specified_type), update_usd=True)

    @staticmethod
    def on_value_changed_callback(attr) -> None:
        node = attr.get_node()
        output_type = attr.get()
        output_attr = node.get_attribute("outputs:samples")
        if output_attr.get_resolved_type().base_type == og.BaseDataType.UNKNOWN and output_type != "":
            if output_type is None:
                raise ValueError(f"Unable to parse type {output_type}")
            output_attr.set_resolved_type(og.Controller.attribute_type(output_type))
