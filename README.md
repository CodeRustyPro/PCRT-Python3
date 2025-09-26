# PCRT (PNG Check & Repair Tool) - Python 3 Version

[![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Version 2.0](https://img.shields.io/badge/Version-2.0-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a modernized, Python 3 compatible version of the original **PCRT (PNG Check & Repair Tool)**.

### Acknowledgment & Credit

This project is a direct port and enhancement of the original tool created by **sherlly**. All credit for the original concept, features, and implementation goes to him.

-   **Original Author:** [sherlly](https://github.com/sherlly)
-   **Original Repository:** [https://github.com/sherlly/PCRT](https://github.com/sherlly/PCRT)

This fork aims to ensure the tool remains functional and accessible for users in modern Python environments.

---

## Description

**PCRT** is a powerful command-line tool to check PNG image integrity and attempt to automatically fix common errors. It's cross-platform and runs on **Windows**, **Linux**, and **macOS**.

### Key Features

* ✅ **Show Image Information:** Display detailed metadata from a PNG's chunks.
* ✅ **Fix PNG Header:** Corrects invalid or missing PNG file signatures.
* ✅ **Fix IHDR CRC:** Recalculates the IHDR checksum, which is often broken by incorrect width/height values.
* ✅ **Fix IDAT CRC:** Corrects checksum errors in image data chunks.
* ✅ **Fix IEND Chunk:** Adds a missing or corrupted IEND chunk.
* ✅ **Extract Appended Data:** Finds and extracts data hidden after the IEND chunk, a common technique for hiding payloads.
* ✅ **Inject Payloads:** Hides a custom payload within the PNG in either ancillary or critical chunks.
* ✅ **Decompress Image Data:** Can decompress raw zlib data from a PNG and attempt to bruteforce its dimensions to recover a viewable image.
* ✅ **Preview Repaired Image:** Automatically opens the repaired image for immediate verification.

---

## Installation

This version requires Python 3.6 or newer.

#### 1. Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/PCRT.git](https://github.com/YOUR_USERNAME/PCRT.git)
cd PCRT