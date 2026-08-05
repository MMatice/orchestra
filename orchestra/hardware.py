"""Detection du materiel disponible.

Le but n'est pas d'etre exhaustif mais de repondre a une seule question :
quelle taille de modele cette machine peut-elle faire tourner confortablement ?
On mesure donc la VRAM exploitable (somme des GPU, ou memoire unifiee sur
Apple Silicon) et on retombe sur la RAM en mode CPU.
"""

from __future__ import annotations

import ctypes
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

_CMD_TIMEOUT_S = 15


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_CMD_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@dataclass
class Gpu:
    name: str
    vram_gb: float
    vendor: str


@dataclass
class Hardware:
    os: str
    arch: str
    ram_gb: float
    gpus: list[Gpu] = field(default_factory=list)

    @property
    def backend(self) -> str:
        if not self.gpus:
            return "cpu"
        return self.gpus[0].vendor

    @property
    def total_vram_gb(self) -> float:
        return round(sum(g.vram_gb for g in self.gpus), 1)

    @property
    def usable_gb(self) -> float:
        """Memoire realistement disponible pour les poids d'un modele.

        Avec GPU : on garde ~10 % de marge (contexte KV, framebuffer, bureau).
        Sans GPU : l'inference CPU tourne sur la RAM, on plafonne a 60 % pour
        laisser vivre le reste du systeme.
        """
        if self.gpus:
            return round(self.total_vram_gb * 0.9, 1)
        return round(self.ram_gb * 0.6, 1)

    def summary(self) -> str:
        if self.gpus:
            gpu_txt = ", ".join(f"{g.name} ({g.vram_gb} Go)" for g in self.gpus)
        else:
            gpu_txt = "aucun GPU detecte"
        return (
            f"{self.os}/{self.arch} | RAM {self.ram_gb} Go | {gpu_txt} | "
            f"backend={self.backend} | budget modele ~{self.usable_gb} Go"
        )


def _detect_nvidia() -> list[Gpu]:
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpus.append(Gpu(parts[0], round(float(parts[1]) / 1024, 1), "cuda"))
        except ValueError:
            continue
    return gpus


def _detect_amd() -> list[Gpu]:
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    gpus = []
    for card, info in data.items():
        for key, value in info.items():
            if "total" in key.lower():
                try:
                    gpus.append(
                        Gpu(card, round(float(value) / (1024**3), 1), "rocm")
                    )
                except (TypeError, ValueError):
                    pass
                break
    return gpus


def _detect_apple(ram_gb: float) -> list[Gpu]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    # Memoire unifiee : Metal peut adresser une large part de la RAM.
    return [Gpu("Apple Silicon (memoire unifiee)", round(ram_gb * 0.7, 1), "metal")]


def _detect_ram_gb() -> float:
    system = platform.system()
    if system == "Windows":
        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 1)
        return 0.0

    if system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024**2), 1)
        except OSError:
            pass
        return 0.0

    if system == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return round(int(out.strip()) / (1024**3), 1)
            except ValueError:
                pass
    return 0.0


def detect() -> Hardware:
    ram_gb = _detect_ram_gb()
    gpus = _detect_nvidia() or _detect_amd() or _detect_apple(ram_gb)
    return Hardware(
        os=platform.system(),
        arch=platform.machine(),
        ram_gb=ram_gb,
        gpus=gpus,
    )
