# Environment for the relocatable pi05 c2t64 W8A8 OpenVINO line.
# Source this, do not execute it.  Everything is resolved from this file's
# directory, so the tree can live anywhere.
PI05_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PI05_ROOT

# OpenVINO device.  A host with one Intel GPU enumerates it as "GPU"; a host
# with several enumerates GPU.0/GPU.1/... and the bare alias then fails
# run_integrated_acc.py's `assert device in core.available_devices`.
export OV_DEVICE="${OV_DEVICE:-GPU}"

PI05_VENV="${PI05_VENV:-$PI05_ROOT/venv}"
export PI05_VENV
export PI05_PYTHON="$PI05_VENV/bin/python"

# The custom GPU plugin must come before the wheel's own libopenvino_intel_gpu_plugin.so.
export LD_LIBRARY_PATH="$PI05_ROOT/custom-plugin-cams-r11:$PI05_VENV/lib/python3.12/site-packages/openvino/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Bundled Level-Zero loader + Intel OpenCL ICD, for hosts whose own stack is
# older than NEO 26.31 / ze_loader 1.32.  Set PI05_USE_BUNDLED_RUNTIME=1 to use
# them; on a host with a current intel-opencl-icd + libze-intel-gpu1 leave it off.
if [ "${PI05_USE_BUNDLED_RUNTIME:-0}" = "1" ]; then
  export LD_LIBRARY_PATH="$PI05_ROOT/runtime/level-zero/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
  export OCL_ICD_VENDORS="$PI05_ROOT/runtime/ocl-vendors"
fi

# plugins.xml carries an absolute path to the plugin .so, so it is generated
# here rather than shipped resolved (the shipped one keeps its provenance hash).
PI05_PLUGINS="$PI05_ROOT/custom-plugin-cams-r11/plugins.local.xml"
export PI05_PLUGINS
cat > "$PI05_PLUGINS" <<XML
<ie>
  <plugins>
    <plugin name="GPU" location="$PI05_ROOT/custom-plugin-cams-r11/libopenvino_intel_gpu_plugin.so"/>
  </plugins>
</ie>
XML
