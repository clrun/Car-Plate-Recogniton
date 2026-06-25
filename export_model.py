from ultralytics import YOLO
import argparse

def export_model(weight_path: str, export_format: str):
    """
    导出YOLO模型为指定格式
    :param weight_path: 训练好的pt权重路径
    :param export_format: 导出格式：onnx/engine/tfjs/torchscript
    """
    model = YOLO(weight_path)

    export_config = {
        "imgsz": 640,
        "simplify": True,
        "opset": 12,
        "batch": 1
    }

    if export_format == "onnx":
        model.export(format="onnx", **export_config)
        print("ONNX 模型导出完成")

    elif export_format == "engine":
        model.export(format="engine", half=True, **export_config)
        print("TensorRT 模型导出完成")

    elif export_format == "tfjs":
        model.export(format="tfjs", **export_config)
        print("TFJS 模型导出完成")

    elif export_format == "torchscript":
        model.export(format="torchscript", **export_config)
        print("TorchScript 模型导出完成")

    else:
        print(f"不支持的导出格式: {export_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True, help="模型权重路径")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "engine", "tfjs", "torchscript"],
                        help="导出格式")
    args = parser.parse_args()

    export_model(args.weight, args.format)