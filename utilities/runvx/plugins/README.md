# RunVX Runtime Plugins

This directory contains optional runtime-loaded plugins for `runvx`. The core `runvx` binary does not link against OpenCV or OpenSSL directly; instead it `dlopen`s these shared libraries when they are available.

## Plugins

| Plugin | Library | Dependency | Features |
|--------|---------|------------|----------|
| Image / Display | `runvx_image_opencv.so` (`runvx_image_opencv.dll` on Windows) | OpenCV 3.4+ | Compressed image read/write, camera/video capture, display, video writer |
| Hash | `runvx_hash_openssl.so` (`runvx_hash_openssl.dll` on Windows) | OpenSSL / libcrypto | MD5 checksums for `compare` GDF command |

## Build behavior

CMake detects OpenCV and OpenSSL at configure time:

* If the dependency is found, the corresponding plugin is built.
* If the dependency is missing, the plugin is skipped and a note is printed; the core `runvx` binary still builds and runs.
* `MIN_DEPS_MODE=ON` disables all plugin builds, producing a minimal `runvx`.

If a dependency becomes available later, the plugin can be rebuilt on the target machine from the same source tree.

## Install location

Plugins are installed to:

```
${CMAKE_INSTALL_PREFIX}/share/mivisionx/runvx_plugins/
```

Runtime plugin shared libraries are placed directly in that directory. Source files, interface headers, and `CMakeLists.txt` for each plugin are placed in per-plugin subdirectories (`image_opencv/`, `hash_openssl/`).

## Runtime search order

1. The `share/mivisionx/runvx_plugins/` directory next to the `runvx` executable
2. Build-tree fallback

## Plugin interface

New plugins can be added by implementing these interfaces and exporting the factory functions `RunVxCreateImagePlugin()` or `RunVxCreateHashPlugin()`. Each plugin folder is self-contained: it contains the interface header, implementation source(s), and its own `CMakeLists.txt`.
