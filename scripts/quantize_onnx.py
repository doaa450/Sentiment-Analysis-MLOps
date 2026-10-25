"""Dynamic INT8 quantization of the student ONNX graph."""

import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

SRC = Path("models/onnx/student_fp32.onnx")
DST = Path("models/onnx/student_int8.onnx")


def main() -> int:
    quantize_dynamic(str(SRC), str(DST), weight_type=QuantType.QInt8)
    fp32, int8 = SRC.stat().st_size, DST.stat().st_size
    sys.stdout.write(f"fp32: {fp32} bytes\nint8: {int8} bytes\nratio: {fp32 / int8:.2f}x\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
