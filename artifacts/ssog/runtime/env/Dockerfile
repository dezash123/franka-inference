# ssog-b580:neo2513 -- the build/run environment for the pi05 B580 campaign on sleepy-joe.
#
# WHY A CONTAINER AT ALL: the host user-space driver on sleepy-joe (and on
# rafael-in-memorium) is NEO 26.31, which cannot compile this campaign's ESIMD translation
# units -- the in-process JIT spins forever and `ocloc` exits 226.  NEO 25.13 builds them in
# ~45 s.  The host kernel driver (`xe`) is fine and is used as-is through /dev/dri.
#
# WHY THIS BASE IMAGE NEEDS NO EXTRA .debs: intel/oneapi-basekit:2025.2.0-0-devel-ubuntu24.04
# already ships exactly the stack h1 measured on --
#   intel-opencl-icd   25.13.33276.22-1133~24.04
#   libze-intel-gpu1   25.13.33276.22-1133~24.04   (libze_intel_gpu.so.1.6.33276 == "1.6.33276+22")
#   libze1 / libze-dev 1.21.1.0-1123~24.04
#   libigdgmm12        22.7.1-1120~24.04
#   oneAPI DPC++       2025.2.0
# so this image only *freezes* that stack and records its provenance.  If the base image is
# ever re-tagged, the build below fails loudly instead of silently running a different driver.
FROM intel/oneapi-basekit:2025.2.0-0-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# 1. Assert the NEO user-space is 25.13.33276.22.  A mismatch is a hard build failure.
ARG NEO_VER=25.13.33276.22-1133~24.04
ARG GMM_VER=22.7.1-1120~24.04
RUN set -eux; \
    for p in intel-opencl-icd libze-intel-gpu1; do \
      v="$(dpkg-query -W -f='${Version}' "$p")"; \
      [ "$v" = "$NEO_VER" ] || { echo "FATAL: $p is $v, expected $NEO_VER" >&2; exit 1; }; \
    done; \
    v="$(dpkg-query -W -f='${Version}' libigdgmm12)"; \
    [ "$v" = "$GMM_VER" ] || { echo "FATAL: libigdgmm12 is $v, expected $GMM_VER" >&2; exit 1; }; \
    test -e /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.6.33276

# 2. Pin it: nothing in this image may upgrade the GPU user-space.
RUN apt-mark hold intel-opencl-icd libze-intel-gpu1 libze1 libze-dev libigdgmm12

# 3. Small conveniences used by the harnesses (npy checks, evidence copy-out).
RUN apt-get update \
 && apt-get install -y --no-install-recommends rsync python3-numpy jq \
 && rm -rf /var/lib/apt/lists/*

# 4. Provenance, readable from any run: `cat /etc/ssog-neo-versions.txt`.
RUN { echo "# ssog-b580:neo2513 built $(date -u +%FT%TZ)"; \
      dpkg-query -W -f='${Package} ${Version}\n' intel-opencl-icd libze-intel-gpu1 libze1 \
                                                 libze-dev libigdgmm12 | sort; \
      echo "icpx $(. /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; icpx --version | head -1)"; \
    } > /etc/ssog-neo-versions.txt \
 && cat /etc/ssog-neo-versions.txt

WORKDIR /work
ENTRYPOINT ["/bin/bash"]
