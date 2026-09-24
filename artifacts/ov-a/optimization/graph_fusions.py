"""Explicit static-shape fusions preserving the saved INT8 weights/scales."""
from collections import defaultdict
import hashlib
import json
import numpy as np
import openvino as ov
from openvino import opset13 as op


def batch_vision(model):
    """Run the identical camera towers in one batch, preserving all images."""
    projects = [n for n in model.get_ordered_ops()
                if n.get_type_name() == "MatMul" and "multi_modal_projector" in n.friendly_name]
    cameras = len(projects)
    assert cameras in (2, 3), cameras
    ends = [n.input_value(0) for n in projects]
    memo, parameters = {}, {}

    def fingerprint(node):
        key = node.get_name()
        if key in memo:
            return memo[key]
        kind = node.get_type_name()
        h = hashlib.sha256()
        h.update(kind.encode())
        h.update(json.dumps(node.get_attributes(), sort_keys=True, default=str).encode())
        if kind == "Parameter":
            parameters[key] = {key: node}
        elif kind == "Constant":
            h.update(node.data.tobytes())
            parameters[key] = {}
        else:
            parameters[key] = {}
            for value in node.input_values():
                parent = value.get_node()
                h.update(fingerprint(parent))
                h.update(str(value.get_index()).encode())
                parameters[key].update(parameters[parent.get_name()])
        memo[key] = h.digest()
        return memo[key]

    digests = [fingerprint(end.get_node()) for end in ends]
    assert len(set(digests)) == 1, "Camera tower computations/weights differ"
    images = []
    for end in ends:
        roots = parameters[end.get_node().get_name()]
        assert len(roots) == 1, roots
        image = next(iter(roots.values()))
        assert list(image.output(0).get_shape()) == [1, 3, 224, 224]
        images.append(image)
    sub = ov.Model([ends[0]], [images[0]]).clone()
    sub.reshape({sub.input(0): [cameras, 3, 224, 224]})
    batched_images = op.concat([image.output(0) for image in images], 0, name="batched_camera_images")
    sub.get_parameters()[0].output(0).replace(batched_images.output(0))
    batched_output = sub.get_results()[0].input_value(0)
    split = op.split(batched_output, op.constant(0, np.int64), cameras, name="batched_camera_features_split")
    for i, end in enumerate(ends):
        end.replace(split.output(i))
    model.validate_nodes_and_infer_types()
    return {
        "camera_count": cameras,
        "identical_tower_fingerprint": digests[0].hex(),
        "batched_output_shape": list(batched_output.get_shape()),
        "camera_parameters": [p.friendly_name for p in images],
        "all_images_preserved": True,
    }


def compressed_weight(node):
    """Match optional Convert(Multiply(Convert(INT8), FP16 scale))."""
    outer_type = None
    if node.get_type_name() == "Convert":
        outer_type = node.get_output_element_type(0)
        node = node.input_value(0).get_node()
    assert node.get_type_name() == "Multiply", node.friendly_name
    left, scale = [node.input_value(i).get_node() for i in range(2)]
    assert left.get_type_name() == "Convert" and scale.get_type_name() == "Constant"
    weight = left.input_value(0).get_node()
    assert weight.get_type_name() == "Constant"
    assert weight.get_output_element_type(0) == ov.Type.i8
    assert list(scale.get_output_shape(0)) == [weight.get_output_shape(0)[0], 1]
    return weight, scale, left.get_output_element_type(0), outer_type


def align_compressed_inner(model, alignment=128, fuse_padding=False):
    """Pad uncommon K sizes with exact zeros so GPU dynamic INT8 is eligible."""
    report, cache = [], {}
    for fc in model.get_ordered_ops():
        if fc.get_type_name() != "MatMul" or ".mlp.fc2/" not in fc.friendly_name:
            continue
        assert fc.get_attributes() == {"transpose_a": False, "transpose_b": True}
        weight, scale, inner_type, outer_type = compressed_weight(fc.input_value(1).get_node())
        n, k = weight.get_output_shape(0)
        padded = ((k + alignment-1)//alignment)*alignment
        if k == padded:
            continue
        key = weight.get_name(), scale.get_name()
        if key not in cache:
            packed = np.pad(weight.data, ((0, 0), (0, padded-k)))
            dequant = op.multiply(op.convert(op.constant(packed), inner_type), scale)
            if outer_type is not None:
                dequant = op.convert(dequant, outer_type)
            cache[key] = dequant
        source = fc.input_value(0)
        if fuse_padding:
            gelu = source.get_node()
            assert gelu.get_type_name() == "Gelu"
            add = gelu.input_value(0).get_node()
            assert add.get_type_name() == "Add"
            up = add.input_value(0).get_node()
            bias = add.input_value(1).get_node()
            assert up.get_type_name() == "MatMul" and bias.get_type_name() == "Constant"
            assert all(len(n.output(0).get_target_inputs()) == 1 for n in (up, add, gelu))
            w1, s1, t1, t2 = compressed_weight(up.input_value(1).get_node())
            padded_w1 = np.pad(w1.data, ((0, padded-k), (0, 0)))
            # New zero-weight rows need a finite nonzero scale.
            padded_s1 = np.pad(s1.data, ((0, padded-k), (0, 0)), constant_values=1)
            dequant = op.multiply(op.convert(op.constant(padded_w1), t1), op.constant(padded_s1))
            if t2 is not None:
                dequant = op.convert(dequant, t2)
            up.input(1).replace_source_output(dequant.output(0))
            padding = [(0, 0)] * bias.data.ndim
            padding[-1] = (0, padded-k)
            add.input(1).replace_source_output(op.constant(np.pad(bias.data, padding)).output(0))
        else:
            rank = len(source.get_shape())
            begin, end = np.zeros(rank, np.int64), np.zeros(rank, np.int64)
            end[-1] = padded-k
            zero = op.convert(op.constant(0, np.float32), source.get_element_type())
            activation = op.pad(source, begin, end, "constant", zero,
                                name=fc.friendly_name + "/align_k")
            fc.input(0).replace_source_output(activation.output(0))
        fc.input(1).replace_source_output(cache[key].output(0))
        report.append({"name": fc.friendly_name, "useful_k": int(k),
                       "padded_k": int(padded), "n": int(n), "weight_scales_preserved": True,
                       "padding_fused_into_fc1": fuse_padding})
    model.validate_nodes_and_infer_types()
    return report


def fuse_qkv(model, scopes=("vision", "language", "expert")):
    groups = defaultdict(list)
    report = []
    cache = {}
    for fc in model.get_ordered_ops():
        if fc.get_type_name() != "MatMul" or ".self_attn." not in fc.friendly_name:
            continue
        if not any(f".{q}_proj/" in fc.friendly_name for q in ("q", "k", "v")):
            continue
        scope = ("vision" if "vision_tower" in fc.friendly_name else
                 "expert" if "gemma_expert" in fc.friendly_name else "language")
        if scope not in scopes:
            continue
        act = fc.input_value(0)
        groups[(act.get_node().get_name(), act.get_index())].append(fc)
    for nodes in groups.values():
        # The final prefix layer only needs K/V for the action expert's cache.
        assert len(nodes) in (2, 3), [(n.friendly_name, str(n.output(0).get_shape())) for n in nodes]
        nodes.sort(key=lambda n: next(i for i, q in enumerate(("q", "k", "v")) if f".{q}_proj/" in n.friendly_name))
        for fc in nodes:
            assert fc.get_attributes() == {"transpose_a": False, "transpose_b": True}
        weights = [compressed_weight(fc.input_value(1).get_node()) for fc in nodes]
        assert len({str(w[2:]) for w in weights}) == 1
        key = tuple((w[0].get_name(), w[1].get_name()) for w in weights)
        if key not in cache:
            # Concatenate original compressed bytes and original scale bytes.
            # No dequantization/requantization or new quantization error here.
            w = op.constant(np.concatenate([w[0].data for w in weights], axis=0))
            s = op.constant(np.concatenate([w[1].data for w in weights], axis=0))
            fused_w = op.multiply(op.convert(w, weights[0][2]), s)
            if weights[0][3] is not None:
                fused_w = op.convert(fused_w, weights[0][3])
            cache[key] = fused_w
        stem = nodes[0].friendly_name.replace(".q_proj/", ".qkv_fused/").replace(".k_proj/", ".qkv_fused/")
        fc = op.matmul(nodes[0].input_value(0), cache[key], False, True, name=stem)
        sizes = [int(n.output(0).get_shape()[-1]) for n in nodes]
        old_outputs = [n.output(0) for n in nodes]
        bias = []
        for n in nodes:
            users = list(n.output(0).get_target_inputs())
            if len(users) != 1 or users[0].get_node().get_type_name() != "Add":
                break
            add = users[0].get_node()
            other = add.input_value(1-users[0].get_index()).get_node()
            if other.get_type_name() != "Constant":
                break
            bias.append((add, other))
        if len(bias) == len(nodes):
            merged_bias = op.constant(np.concatenate([b.data for _, b in bias], axis=-1))
            fc = op.add(fc, merged_bias, name=stem + "/bias")
            old_outputs = [a.output(0) for a, _ in bias]
        split = op.variadic_split(fc, op.constant(-1, np.int64), op.constant(sizes, np.int64),
                                 name=stem + "/split")
        for i, old in enumerate(old_outputs):
            split.output(i).get_tensor().set_names(old.get_names())
            old.replace(split.output(i))
        report.append({
            "fused": stem, "originals": [n.friendly_name for n in nodes],
            "activation_shape": list(nodes[0].input_value(0).get_shape()),
            "output_sizes": sizes, "bias_fused": len(bias) == len(nodes),
            "preserved_compressed_weights_and_scales": True,
        })
    model.validate_nodes_and_infer_types()
    return report
