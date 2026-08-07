import os
import sys
import numpy as np

def convert_onnx_folder(source_folder):
    """
    Recursively scans source_folder for all best.onnx / *.onnx files
    and converts them to best_net.mat files in the same directory.
    """
    try:
        import onnx
        from onnx import numpy_helper
        import scipy.io as sio
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        print("Please ensure 'onnx', 'numpy', and 'scipy' are installed in your Python environment.")
        return

    if not os.path.exists(source_folder):
        print(f"Directory not found: {source_folder}")
        return

    print(f"Scanning for ONNX models in: {source_folder}")
    converted_count = 0

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".onnx"):
                onnx_path = os.path.join(root, file)
                mat_filename = file.replace(".onnx", "_net.mat")
                if file == "best.onnx":
                    mat_filename = "best_net.mat"
                
                mat_path = os.path.join(root, mat_filename)
                print(f"\n----------------------------------------")
                print(f"Processing: {onnx_path}")
                try:
                    model = onnx.load(onnx_path)
                    weights_dict = {}
                    
                    # Extract initializers (weight tensors and biases)
                    for init in model.graph.initializer:
                        W = numpy_helper.to_array(init)
                        # Sanitize key names for MATLAB struct field compatibility
                        clean_key = init.name.replace('.', '_').replace('/', '_').replace(':', '_').replace('-', '_')
                        weights_dict[clean_key] = W
                        print(f"  Extracted Weight '{clean_key}': shape={W.shape}, dtype={W.dtype}")

                    # Extract graph layer node descriptions
                    layers_info = []
                    for idx, node in enumerate(model.graph.node):
                        layers_info.append(f"Layer_{idx}_{node.op_type}")

                    sio.savemat(mat_path, {
                        'onnx_weights': weights_dict,
                        'layer_names': layers_info,
                        'source_onnx': file
                    })
                    print(f"Successfully created MAT file: {mat_path}")
                    converted_count += 1
                except Exception as ex:
                    print(f"Failed to convert {onnx_path}: {ex}")

    print(f"\n========================================")
    print(f"Conversion complete! Converted {converted_count} ONNX model(s) to .mat format.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(script_dir, "single_source_trained_model", "DUR100ns_2p18G_600km_70deg_r15km_20to30mps")

    convert_onnx_folder(target_dir)
