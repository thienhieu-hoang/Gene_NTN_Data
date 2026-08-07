import os
import sys
import numpy as np

base_dir = r"c:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Gene_NTN_Data\MATLAB\NTN_thruput\BER_cal\single_source_trained_model\DUR100ns_2p18G_600km_70deg_r15km_20to30mps"

try:
    import onnx
    from onnx import numpy_helper
    import scipy.io as sio

    def inspect_and_convert(subfolder):
        onnx_file = os.path.join(base_dir, subfolder, "results", "best.onnx")
        mat_file  = os.path.join(base_dir, subfolder, "results", "best_net.mat")
        
        if not os.path.exists(onnx_file):
            print(f"Skipping {onnx_file} (not found)")
            return

        print(f"\n==========================================")
        print(f"Converting: {onnx_file}")
        model = onnx.load(onnx_file)
        graph = model.graph
        
        print("Inputs:")
        for inp in graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  {inp.name}: shape={shape}, type={inp.type.tensor_type.elem_type}")
            
        print("Outputs:")
        for out in graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  {out.name}: shape={shape}, type={out.type.tensor_type.elem_type}")

        print("\nNodes (Layers):")
        nodes_info = []
        for i, node in enumerate(graph.node):
            print(f"  Node {i}: {node.op_type} (name='{node.name}', inputs={node.input}, outputs={node.output})")
            nodes_info.append({
                'op_type': node.op_type,
                'name': node.name,
                'inputs': list(node.input),
                'outputs': list(node.output)
            })

        weights_dict = {}
        for init in graph.initializer:
            W = numpy_helper.to_array(init)
            key = init.name.replace('.', '_').replace('/', '_').replace(':', '_').replace('-', '_')
            weights_dict[key] = W
            print(f"  Initializer {key}: shape={W.shape}, dtype={W.dtype}")

        sio.savemat(mat_file, {
            'weights': weights_dict,
            'info': f"Converted from {onnx_file}"
        })
        print(f"Saved MAT file to: {mat_file}")

    inspect_and_convert("LS_-5")
    inspect_and_convert("LI_-5")

except Exception as e:
    print(f"Error: {e}")
