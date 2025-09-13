# CUDA Study Project
![CUDA Logo](https://upload.wikimedia.org/wikipedia/commons/b/b9/Nvidia_CUDA_Logo.jpg)

A study project in C++ using **CUDA** for NVIDIA GPUs. This project explores basic GPU programming concepts, including:

- Vector operations
- Matrix operations
- Image handling (loading, processing, saving)

The project is designed for learning and experimentation. In the future, more examples and advanced techniques will be added.

---

## Requirements

- **C++ compiler** (supports C++11 or later)  
- **CUDA Toolkit** (tested with CUDA 12.x)  
- **NVIDIA GPU** with CUDA support  
- **stb_image / stb_image_write** (included or downloaded from [https://github.com/nothings/stb](https://github.com/nothings/stb))  
- **CMake** (optional, for building a more structured project)

---

## Compilation

If you are compiling manually with `nvcc`:

```bash
nvcc main.cu -o cuda_project
