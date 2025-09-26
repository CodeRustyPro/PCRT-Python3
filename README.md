# PCRT (PNG Check & Repair Tool) - Python 3 Version

[![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Version 2.0](https://img.shields.io/badge/Version-2.0-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modernized, Python 3 compatible version of the original **PCRT (PNG Check & Repair Tool)**.

### Acknowledgment

This project is a port and enhancement of the original tool by **sherlly**. Full credit for the original concept and features goes to the [original repository](https://github.com/sherlly/PCRT).

---

## Key Features

* ✅ Fix common PNG errors (Header, IHDR/IDAT CRCs, IEND chunk).
* ✅ Display detailed image metadata.
* ✅ Extract data appended after the `IEND` chunk.
* ✅ Inject text payloads into image chunks (steganography).
* ✅ Decompress raw image data and bruteforce dimensions.
* ✅ Automatically preview the repaired image.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/coderustypro/PCRT-Python3.git](https://github.com/coderustypro/PCRT-Python3.git)
    cd PCRT-Python3
    ```
2.  **Run the tool:**
    The script requires the `Pillow` library. If it fails, run `pip install Pillow`.
    ```bash
    python3 PCRT.py --help
    ```

---

## Usage


**Check and repair a broken image:**
```bash
python3 PCRT.py -i secret_image.png
```
\
## Example: Before & After Repair

Here is an example of a corrupted image (`secret_image.png`) being repaired into `output.png`.

| Before (Corrupt) | After (Repaired) |
| :---: | :---: |
| ![Corrupt Input Image](https://github.com/CodeRustyPro/PCRT-Python3/blob/main/secret_image.png) | ![Repaired Output Image](https://github.com/CodeRustyPro/PCRT-Python3/blob/main/output.png) |