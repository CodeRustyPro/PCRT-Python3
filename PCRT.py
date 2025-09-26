#!/usr/bin/env python3
# -*- coding:utf-8 -*-

__author__ = "sherlly"
__version__ = "2.0"

import zlib
import struct
import re
import os
import argparse
import itertools
import platform
import sys
import binascii
import subprocess

# Platform-specific color setup for Windows
if platform.system() == "Windows":
    import ctypes
    STD_OUTPUT_HANDLE = -11
    FOREGROUND_BLUE = 0x09
    FOREGROUND_GREEN = 0x0a
    FOREGROUND_RED = 0x0c
    FOREGROUND_SKYBLUE = 0x0b
    std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

    def set_cmd_text_color(color, handle=std_out_handle):
        return ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)

    def resetColor():
        set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)

    def printRed(msg):
        set_cmd_text_color(FOREGROUND_RED)
        print(msg, end='')
        resetColor()

    def printSkyBlue(msg):
        set_cmd_text_color(FOREGROUND_SKYBLUE)
        print(msg, end='')
        resetColor()

    def printGreen(msg):
        set_cmd_text_color(FOREGROUND_GREEN)
        print(msg, end='')
        resetColor()

def str2hex(s: bytes) -> str:
    """Converts bytes to an uppercase hex string."""
    return binascii.hexlify(s).upper().decode('ascii')

def int2hex(i: int) -> str:
    """Converts an integer to a standard hex string like '0xAB'."""
    return f'0x{i:X}'

def str2num(s: bytes, n: int = 0) -> int:
    """Converts bytes to a number. n=4 for 4-byte big-endian unsigned int."""
    if n == 4:
        return struct.unpack('!I', s)[0]
    else:
        return int.from_bytes(s, 'big')

def WriteFile(filename: str):
    """Opens a file for writing in binary mode, overwriting if it exists."""
    if os.path.isfile(filename):
        os.remove(filename)
    return open(filename, 'wb+')

def ReadFile(filename: str) -> bytes:
    """Reads a file in binary mode and returns its content."""
    try:
        with open(filename, 'rb') as file:
            data = file.read()
    except IOError as e:
        print(Termcolor('Error', f'{e}: {filename}'))
        return b''
    return data

def Termcolor(flag: str, sentence: str) -> str:
    """Formats a message with colors for the terminal."""
    system = platform.system()
    if system == 'Linux' or system == 'Darwin':
        color_map = {
            'Notice': "\033[0;34m",
            'Detected': "\033[0;32m",
        }
        error_color = "\033[0;31m"
        reset_color = "\033[0m"
        color = color_map.get(flag, error_color)
        return f"{color}[{flag}]{reset_color} {sentence}"
    elif system == 'Windows':
        if flag == 'Notice':
            printSkyBlue(f'[{flag}] ')
        elif flag == 'Detected':
            printGreen(f'[{flag}] ')
        elif flag in ('Error', 'Warning', 'Failed'):
            printRed(f'[{flag}] ')
        return sentence
    else:
        return f"[{flag}] {sentence}"

def open_path(path: str):
    """Opens a file or directory in the default application."""
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux and other Unix-like
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print(Termcolor('Error', f"Could not open path '{path}': {e}"))


class PNG(object):
    def __init__(self, in_file='', out_file='output.png', choices='', mode=0):
        self.in_file = in_file
        self.out_file = out_file
        self.choices = choices
        self.i_mode = mode
        self.file = None
        self.width = 0
        self.height = 0
        self.bits = 0
        self.mode = 0
        self.compression = 0
        self.filter = 0
        self.interlace = 0
        self.channel = 0
        self.content = {}

    def __del__(self):
        if self.file:
            self.file.close()

    def AddPayload(self, name, payload, way):
        data = self.LoadPNG()
        if not data:
            return -1
        self.file = WriteFile(self.out_file)
        if way == 1:
            # way1:add ancillary chunk
            payload_chunk = self.MakeAncillary(name, payload.encode('utf-8'))
            pos = data.find(b'IHDR')
            self.file.write(data[:pos+21])
            self.file.write(payload_chunk)
            self.file.write(data[pos+21:])
        elif way == 2:
            # way2:add critical chunk:IDAT
            name = 'IDAT'
            payload_chunk = self.MakeCritical(name, payload.encode('utf-8'))
            pos = data.find(b'IEND')
            self.file.write(data[:pos-4])
            self.file.write(payload_chunk)
            self.file.write(data[pos-4:])
        print(Termcolor('Detected', f"Payload added to {self.out_file}"))

    def MakeCritical(self, name, payload):
        print(Termcolor('Notice', f'Payload chunk name: {name}'))
        payload = zlib.compress(payload)
        length = len(payload)
        name_bytes = name.encode('ascii')
        crc = zlib.crc32(name_bytes + payload) & 0xffffffff
        return struct.pack(f'!I4s{length}sI', length, name_bytes, payload, crc)

    def MakeAncillary(self, name, payload):
        if name is None:
            name = self.RanAncillaryName()
        name = name[0].lower() + name[1:4].upper()
        print(Termcolor('Notice', f'Payload chunk name: {name}'))
        length = len(payload)
        name_bytes = name.encode('ascii')
        crc = zlib.crc32(name_bytes + payload) & 0xffffffff
        return struct.pack(f'!I4s{length}sI', length, name_bytes, payload, crc)

    def RanAncillaryName(self):
        import random, string
        return ''.join(random.sample(string.ascii_lowercase, 4))

    def GetPicInfo(self, ihdr=b''):
        data = self.LoadPNG()
        if not data:
            return -1
        if not ihdr:
            pos, IHDR_chunk = self.FindIHDR(data)
            if pos == -1:
                print(Termcolor('Detected', 'Lost IHDR chunk'))
                return -1
            length = struct.unpack('!I', IHDR_chunk[:4])[0]
            ihdr = IHDR_chunk[8:8 + length]

        self.width, self.height, self.bits, self.mode, self.compression, self.filter, self.interlace = struct.unpack('!IIBBBBB', ihdr)

        if self.mode in (0, 3): self.channel = 1
        elif self.mode == 2: self.channel = 3
        elif self.mode == 4: self.channel = 2
        elif self.mode == 6: self.channel = 4
        else: self.channel = 0
        
        self.content = self.FindAncillary(data)
        return 0

    def PrintPicInfo(self):
        if self.GetPicInfo() == -1:
            return -1

        mode_dict = {0: 'Grayscale', 2: 'RGB', 3: 'Indexed', 4: 'Grayscale with Alpha', 6: 'RGB with Alpha'}
        compress_dict = {0: 'Deflate'}
        filter_dict = {0: 'None', 1: 'Sub', 2: 'Up', 3: 'Average', 4: 'Paeth'}
        interlace_dict = {0: 'Noninterlaced', 1: 'Adam7 interlaced'}

        print('\n-------------------------Image Information---------------------------------------')
        print(f'Image Width: {self.width}\nImage Height: {self.height}\nBit Depth: {self.bits}\nChannel: {self.channel}')
        print(f'ColorType: {mode_dict.get(self.mode, "Unknown")}')
        print(f'Interlace: {interlace_dict.get(self.interlace, "Unknown")}\nFilter method: {filter_dict.get(self.filter, "Unknown")}\nCompression method: {compress_dict.get(self.compression, "Unknown")}')
        print('Content: ')
        for k, v in self.content.items():
            if v:
                full_text = b'\n'.join(v).decode('latin-1')
                cleaned_text = "\n".join(line for line in full_text.splitlines() if line.strip())
                print(f'{k}: \n{cleaned_text}')
        print('--------------------------------------------------------------------------------\n')

    def ClearFilter(self, idat: bytes, width: int, height: int, channel: int, bits: int = 8) -> bytes:
        bpp = max(1, channel * bits // 8)
        recon_data = bytearray()
        prior_line = bytearray(width * bpp)

        scanline_len = width * bpp
        filtered_scanline_len = scanline_len + 1
        
        if len(idat) % filtered_scanline_len != 0:
             print(Termcolor('Warning', f'Decompressed IDAT length ({len(idat)}) is not a multiple of scanline size ({filtered_scanline_len}).'))

        for i in range(height):
            start = i * filtered_scanline_len
            if start >= len(idat): break
            
            filter_type = idat[start]
            scanline = idat[start + 1: start + 1 + scanline_len]
            recon_line = bytearray(scanline_len)

            if filter_type == 0: recon_line[:] = scanline
            elif filter_type == 1: # Sub
                for x in range(scanline_len):
                    recon_a = recon_line[x - bpp] if x >= bpp else 0
                    recon_line[x] = (scanline[x] + recon_a) & 0xFF
            elif filter_type == 2: # Up
                for x in range(scanline_len):
                    recon_b = prior_line[x]
                    recon_line[x] = (scanline[x] + recon_b) & 0xFF
            elif filter_type == 3: # Average
                for x in range(scanline_len):
                    recon_a = recon_line[x - bpp] if x >= bpp else 0
                    recon_b = prior_line[x]
                    recon_line[x] = (scanline[x] + (recon_a + recon_b) // 2) & 0xFF
            elif filter_type == 4: # Paeth
                def predictor(a, b, c):
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                for x in range(scanline_len):
                    a = recon_line[x - bpp] if x >= bpp else 0
                    b = prior_line[x]
                    c = prior_line[x - bpp] if x >= bpp else 0
                    recon_line[x] = (scanline[x] + predictor(a, b, c)) & 0xFF
            else:
                print(Termcolor('Error', f'Unknown filter type {filter_type} encountered.'))
                return b''

            recon_data.extend(recon_line)
            prior_line = recon_line
        return bytes(recon_data)

    def LoadPNG(self) -> bytes:
        data = ReadFile(self.in_file)
        if not data:
            return b''
        if not self.CheckFormat(data):
            print(Termcolor('Warning', 'The file may not be a valid PNG image.'))
            return b''
        return data

    def DecompressPNG(self, data, width=0, height=0):
        try:
            from PIL import Image
        except ImportError as e:
            print(Termcolor('Error', str(e)))
            print("Try 'pip install Pillow' to use this feature.")
            return -1

        idat_data = b''.join(data) if isinstance(data, list) else data
        try:
            z_idat = zlib.decompress(idat_data)
        except zlib.error as e:
            print(Termcolor('Error', f"Zlib decompression failed: {e}"))
            return -1
        
        length = len(z_idat)

        if width == 0 and height == 0: # Bruteforce mode
            import shutil
            channel_dict = {1: 'L', 3: 'RGB', 2: 'LA', 4: 'RGBA'}
            PATH = 'tmp/'
            if os.path.isdir(PATH): shutil.rmtree(PATH)
            os.mkdir(PATH)

            for bits in [8, 16]:
                for channel in [4, 3, 1, 2]:
                    bpp = channel * bits // 8
                    if bpp == 0: continue
                    for i in range(1, int(length**0.5) + 1):
                        if (length % i == 0):
                            h, w = i, length // i
                            # Check combinations for (width, height)
                            for width_cand, height_cand in [(w, h), (h, w)]:
                                if (width_cand * bpp + 1) == 0: continue
                                if height_cand * (width_cand * bpp + 1) == length:
                                    tmp = self.ClearFilter(z_idat, width_cand, height_cand, channel, bits)
                                    if tmp:
                                        img = Image.frombytes(channel_dict[channel], (width_cand, height_cand), tmp)
                                        filename = os.path.join(PATH, f'test({width_cand}x{height_cand})_{bits}bits_{channel}channel.png')
                                        img.save(filename)
            
            print(Termcolor('Notice', f"Bruteforce generated possible images in '{PATH}' directory."))
            open_path(os.path.abspath(PATH))
            size_str = input('Input width, height, bits, and channel (space-separated): ')
            shutil.rmtree(PATH)
            
            try:
                size = [int(s) for s in size_str.split()]
                width, height, bits, channel = size
                tmp = self.ClearFilter(z_idat, width, height, channel, bits)
                if tmp:
                    img = Image.frombytes(channel_dict[channel], (width, height), tmp)
                    img.save('decompress.png')
                    img.show()
            except (ValueError, IndexError):
                print(Termcolor('Error', 'Invalid input.'))
                return -1
        else:
            # Load from PNG config if not specified
            if width == 1 and height == 1:
                if self.GetPicInfo() == -1: return -1
                width, height, channel, bits = self.width, self.height, self.channel, self.bits
            
            unfiltered_idat = self.ClearFilter(z_idat, width, height, channel, bits)
            mode_dict = {0: 'L', 2: 'RGB', 3: 'P', 4: 'LA', 6: 'RGBA'}
            if self.mode in mode_dict and unfiltered_idat:
                img = Image.frombytes(mode_dict[self.mode], (width, height), unfiltered_idat)
                img.show()
                img.save('zlib.png')
        return 0

    def FindAncillary(self, data: bytes) -> dict:
        attach_txt = [b'eXIf', b'iTXt', b'tEXt', b'zTXt']
        content = {k.decode('ascii'): [] for k in attach_txt}
        for text_type in attach_txt:
            pos = 0
            while True:
                pos = data.find(text_type, pos)
                if pos == -1: break
                try:
                    length = str2num(data[pos - 4:pos], n=4)
                    content[text_type.decode('ascii')].append(data[pos + 4:pos + 4 + length])
                except (struct.error, IndexError):
                    pass # Ignore malformed chunks
                pos += 1
        return content

    def CheckPNG(self):
        data = self.LoadPNG()
        if not data: return -1

        self.file = WriteFile(self.out_file)
        if self.CheckHeader(data) == -1: return -1
        if self.CheckIHDR(data) == -1: return -1
        res, _ = self.CheckIDAT(data)
        if res == -1: return -1
        self.CheckIEND(data)
        print(Termcolor('Notice', 'PNG check complete.'))
        self.file.close()

        choice = self.choices or input(Termcolor('Notice', 'Show the repaired image? (y/n) [default: n] ')).lower()
        if choice == 'y':
            try:
                from PIL import Image
                img = Image.open(self.out_file)
                img.show()
            except ImportError as e:
                print(Termcolor('Error', str(e)))
                print("Try 'pip install Pillow' to use this feature.")
        return 0

    def Checkcrc(self, chunk_type: bytes, chunk_data: bytes, checksum: bytes) -> bytes or None:
        calc_crc_int = zlib.crc32(chunk_type + chunk_data) & 0xffffffff
        calc_crc_bytes = struct.pack('!I', calc_crc_int)
        return calc_crc_bytes if calc_crc_bytes != checksum else None

    def CheckFormat(self, data: bytes) -> bool:
        return all(feature in data for feature in [b'PNG', b'IHDR', b'IDAT', b'IEND'])

    def CheckHeader(self, data: bytes) -> int:
        header = data[:8]
        correct_header = binascii.unhexlify('89504E470D0A1A0A')
        if header != correct_header:
            print(Termcolor('Detected', 'Wrong PNG header!'))
            print(f'File header: {str2hex(header)}\nCorrect header: {str2hex(correct_header)}')
            choice = self.choices or input(Termcolor('Notice', 'Auto fix? (y/n) [default: y] ')).lower()
            if choice in ('y', ''):
                header = correct_header
                print(Termcolor('Notice', f'Header fixed: {str2hex(header)}'))
            else:
                return -1
        else:
            print(Termcolor('Notice', 'Correct PNG header.'))
        self.file.write(header)
        return 0

    def FindIHDR(self, data: bytes) -> tuple:
        pos = data.find(b'IHDR')
        if pos == -1: return -1, b''
        idat_begin = data.find(b'IDAT')
        end_pos = idat_begin - 4 if idat_begin != -1 else pos + 21
        IHDR_chunk = data[pos - 4: end_pos]
        return pos, IHDR_chunk

    def CheckIHDR(self, data: bytes) -> int:
        pos, IHDR_chunk = self.FindIHDR(data)
        if pos == -1:
            print(Termcolor('Detected', 'Lost IHDR chunk.'))
            return -1
        try:
            length = struct.unpack('!I', IHDR_chunk[:4])[0]
            chunk_type = IHDR_chunk[4:8]
            chunk_ihdr = IHDR_chunk[8:8 + length]
            width, height = struct.unpack('!II', chunk_ihdr[:8])
            crc = IHDR_chunk[8 + length:12 + length]
        except (struct.error, IndexError):
            print(Termcolor('Error', 'Malformed IHDR chunk.'))
            return -1
        
        calc_crc = self.Checkcrc(chunk_type, chunk_ihdr, crc)
        if calc_crc:
            print(Termcolor('Detected', f'Error IHDR CRC at offset {int2hex(pos+8+length)}'))
            print(f'Chunk CRC: {str2hex(crc)}\nCorrect CRC: {str2hex(calc_crc)}')
            choice = self.choices or input(Termcolor('Notice', 'Try fixing it? (y/n) [default: y] ')).lower()
            if choice in ('y', ''):
                IHDR_chunk = IHDR_chunk[:8+length] + calc_crc
                print(Termcolor('Notice', 'IHDR CRC fixed.'))
        else:
            print(Termcolor('Notice', f'Correct IHDR CRC at offset {int2hex(pos+8+length)}'))
        
        self.file.write(IHDR_chunk)
        print(Termcolor('Notice', f'IHDR chunk check complete at offset {int2hex(pos-4)}.'))
        self.GetPicInfo(ihdr=chunk_ihdr)
        return 0

    def CheckIDAT(self, data: bytes) -> tuple:
        idat_begin = data.find(b'IDAT') - 4
        if idat_begin == -1:
            print(Termcolor('Detected', 'Lost all IDAT chunks!'))
            return -1, []
        
        pos_IEND = data.find(b'IEND')
        idat_chunks = []
        pos_list = [g.start() for g in re.finditer(b'IDAT', data) if pos_IEND == -1 or g.start() < pos_IEND]

        for i, start_pos in enumerate(pos_list):
            chunk_start = start_pos - 4
            if i + 1 < len(pos_list):
                chunk_end = pos_list[i+1] - 4
            else:
                chunk_end = pos_IEND - 4 if pos_IEND != -1 else -12
            
            idat_chunks.append(data[chunk_start:chunk_end])

        offset = idat_begin
        idat_data_table = []
        for chunk in idat_chunks:
            try:
                length = struct.unpack('!I', chunk[:4])[0]
                chunk_type = chunk[4:8]
                chunk_data = chunk[8:8 + length]
                crc = chunk[8 + length:12 + length]
            except (struct.error, IndexError):
                print(Termcolor('Warning', f'Skipping malformed IDAT chunk at offset {int2hex(offset)}.'))
                offset += len(chunk)
                continue
            
            if length != len(chunk_data):
                print(Termcolor('Detected', f'Error IDAT data length at offset {int2hex(offset)}'))
                # Simplified: No longer attempting complex DOS2Unix fix
            else:
                print(Termcolor('Notice', f'Correct IDAT length at offset {int2hex(offset)}'))
            
            calc_crc = self.Checkcrc(chunk_type, chunk_data, crc)
            if calc_crc:
                print(Termcolor('Detected', f'Error IDAT CRC at offset {int2hex(offset+8+length)}'))
                choice = self.choices or input(Termcolor('Notice', 'Try fixing it? (y/n) [default: y] ')).lower()
                if choice in ('y', ''):
                    chunk = chunk[:8+length] + calc_crc
                    print(Termcolor('Notice', 'IDAT CRC fixed.'))
            else:
                print(Termcolor('Notice', f'Correct IDAT CRC at offset {int2hex(offset+8+length)}'))
            
            self.file.write(chunk)
            idat_data_table.append(chunk_data)
            offset += len(chunk)
        
        print(Termcolor('Notice', f'IDAT chunk check complete starting at offset {int2hex(idat_begin)}.'))
        return 0, idat_data_table

    def CheckIEND(self, data: bytes):
        standard_IEND = b'\x00\x00\x00\x00IEND\xaeB`\x82'
        pos = data.find(b'IEND')
        if pos == -1:
            print(Termcolor('Detected', 'Lost IEND chunk! Auto fixing...'))
            IEND = standard_IEND
        else:
            IEND = data[pos - 4:pos + 8]
            if IEND != standard_IEND:
                print(Termcolor('Detected', 'Error IEND chunk! Auto fixing...'))
                IEND = standard_IEND
            else:
                print(Termcolor('Notice', 'Correct IEND chunk.'))
            
            appended_data = data[pos+8:]
            if appended_data:
                print(Termcolor('Detected', f'Data (length: {len(appended_data)}) found after IEND chunk.'))
                while True:
                    choice = input(Termcolor('Notice', 'Extract data to: <1> File <2> Terminal <3> Quit [default: 3] '))
                    if choice == '1':
                        filename = input('Enter filename to save appended data: ')
                        with open(filename, 'wb') as f:
                            f.write(appended_data)
                        print(Termcolor('Notice', f"Data written to {filename}."))
                        open_path(os.path.abspath(os.path.dirname(filename) or '.'))
                    elif choice == '2':
                        print('Data (hex):', str2hex(appended_data))
                        try:
                            print('Data (text):', appended_data.decode('utf-8', errors='ignore'))
                        except: pass
                    else: break
        self.file.write(IEND)


if __name__ == '__main__':
    banner = f"""
     ____   ____ ____ _____ 
    |  _ \ / ___|  _ \_   _|
    | |_) | |   | |_) || |  
    |  __/| |___|  _ < | |  
    |_|    \____|_| \_\|_|  

    PNG Check & Repair Tool 

Project address: https://github.com/sherlly/PCRT
Author: sherlly
Version: {__version__}
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help="don't show the banner information")
    parser.add_argument('-y', '--yes', help='auto choose yes for fixes', action='store_true')
    parser.add_argument('-m', '--message', help='show the image information', action='store_true')
    parser.add_argument('-n', '--name', help="payload chunk name [Default: random]")
    parser.add_argument('-p', '--payload', help="payload string to hide in the image")
    parser.add_argument('-w', '--way', type=int, default=1, choices=[1, 2], help="payload chunk type: [1] ancillary, [2] critical [Default: 1]")
    parser.add_argument('-d', '--decompress', help="decompress a raw zlib data file and attempt to bruteforce dimensions")
    parser.add_argument('-i', '--input', help='Input PNG file name')
    parser.add_argument('-f', '--file', help='Select input PNG file from a file dialog window', action='store_true')
    parser.add_argument('-o', '--output', default='output.png', help='Output repaired file name [Default: output.png]')
    args = parser.parse_args()

    if not args.quiet:
        print(banner)

    choices = 'y' if args.yes else ''
    my_png = None

    if args.decompress:
        z_data = ReadFile(args.decompress)
        if z_data:
            my_png = PNG()
            my_png.DecompressPNG([z_data], width=0, height=0)
    else:
        in_file = args.input
        if args.file:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                in_file = filedialog.askopenfilename(title="Select a PNG file", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            except ImportError as e:
                print(Termcolor('Error', str(e)))
                print("Python's tkinter module is required for the file dialog. Try `sudo apt-get install python3-tk` or similar.")
                sys.exit(1)
        
        if in_file:
            my_png = PNG(in_file, args.output, choices=choices)
            if args.message:
                my_png.PrintPicInfo()
            elif args.payload:
                my_png.AddPayload(args.name, args.payload, args.way)
            else:
                my_png.CheckPNG()
        else:
            if not any(arg in sys.argv for arg in ['-h', '--help']):
                 print(Termcolor('Notice', "No input file provided."))
            parser.print_help()