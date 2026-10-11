"""Size-gated optional DeepMIMO acquisition. No automatic large downloads.

The archive stays compressed. Conversion specializes the official DeepMIMO
equations to an isotropic 8x1 BS ULA, one UE antenna, and subcarrier zero.
"""

from __future__ import annotations
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.parse import urljoin, urlencode
from zipfile import ZipFile
import numpy as np
from scipy.io import loadmat
from signal_rescue.channels import DOWNLOAD_CAP, PROCESSED_CAP, keyed_seed

OFFICIAL = "https://deepmimo.net"


def acquire(scenario, destination, *, cap=DOWNLOAD_CAP):
    if not scenario.replace("_", "").isalnum() or not 0 < cap <= DOWNLOAD_CAP:
        raise ValueError("Invalid scenario or download cap")
    destination = Path(destination)
    if destination.exists() or destination.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Will not overwrite {destination}")
    url = OFFICIAL + "/api/download/secure?" + urlencode({"filename": scenario + ".zip"})
    headers = {"User-Agent": "DeepMIMO-Python/4.0", "Accept": "*/*"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        metadata = json.loads(response.read(64_000))
    size = metadata.get("fileSize")
    if not isinstance(size, int) or not 0 < size <= cap:
        raise ValueError(f"Archive size {size!r} is missing or exceeds {cap} bytes; not downloaded")
    redirect = urljoin(OFFICIAL, metadata["redirectUrl"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    created = False
    try:
        with urllib.request.urlopen(urllib.request.Request(redirect, headers=headers), timeout=30) as response:
            header_size = response.headers.get("Content-Length")
            if header_size and int(header_size) > cap:
                raise ValueError("Download response exceeds cap")
            with destination.open("xb") as output:
                created = True
                while True:
                    chunk = response.read(min(64_000, cap - downloaded + 1))
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > cap:
                        raise ValueError("Streaming byte cap reached; partial archive discarded")
                    output.write(chunk)
        if downloaded != size:
            raise ValueError("Download length differs from the official metadata")
        with ZipFile(destination) as archive:
            listing = [{"name": entry.filename, "bytes": entry.file_size}
                       for entry in archive.infolist()]
    except Exception:
        # Only the newly-created, exact target is removed, never an existing file.
        if created and destination.is_file():
            destination.unlink()
        raise
    manifest = {"scenario": scenario, "source_url": url, "download_bytes": downloaded,
                "download_cap_bytes": cap, "archive_members": listing}
    destination.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def synthesize_ula(power_dbw, phase_degrees, polar_degrees, azimuth_degrees, delay):
    """DeepMIMO's no-LPF, k=0, N=512 specialization (no angle-axis relabeling).

    Panel [8,1] lies on y; the reciprocal uplink is the downlink transpose,
    NOT the Hermitian transpose. All users use the SAME frequency bin.
    """
    valid = (np.isfinite(power_dbw) & np.isfinite(phase_degrees) &
             np.isfinite(polar_degrees) & np.isfinite(azimuth_degrees) &
             np.isfinite(delay) & (delay < 512 / 10e6))
    power = np.where(valid, 10 ** (np.nan_to_num(power_dbw, nan=-300) / 10), 0)
    theta = np.deg2rad(np.nan_to_num(polar_degrees))
    phi = np.deg2rad(np.nan_to_num(azimuth_degrees))
    phase = np.deg2rad(np.nan_to_num(phase_degrees))
    response = np.exp(1j * np.pi * np.arange(8)[None, :, None]
                      * (np.sin(theta) * np.sin(phi))[:, None, :])
    gains = np.sqrt(power / 512) * np.exp(1j * phase)
    return np.sum(response * gains[:, None, :], axis=2).astype(np.complex64)


def convert_slice(archive_path, output_path, *, tile_metres=64, guard_metres=8,
                  realizations_per_group=3, max_groups=80):
    archive_path, output_path = Path(archive_path), Path(output_path)
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError("Will not overwrite an existing channel slice or manifest")
    if archive_path.stat().st_size > DOWNLOAD_CAP:
        raise ValueError("Source archive exceeds the 100 MB cap")
    with ZipFile(archive_path) as archive:
        def read_array(name):
            member = f"{name}_t001_tx000_r000.mat"
            if archive.getinfo(member).file_size > DOWNLOAD_CAP:
                raise ValueError("Source member exceeds the 100 MB cap")
            # Read only the required MAT member, never extract interactions/geometry.
            with archive.open(member) as source:
                return loadmat(source)[name]

        positions = read_array("rx_pos")
        powers = read_array("power")
        tiles = np.floor(positions[:, :2] / tile_metres).astype(int)
        within = positions[:, :2] - tiles * tile_metres
        retained = np.all((within >= guard_metres) &
                          (within < tile_metres - guard_metres), axis=1)
        retained &= np.any(np.isfinite(powers), axis=1)
        selected, group_ids = [], []
        for tile in sorted(set(map(tuple, tiles[retained]))):
            rows = np.flatnonzero(retained & np.all(tiles == tile, axis=1))
            if len(rows) < 12 * realizations_per_group:
                continue
            group = f"tile-{tile[0]}-{tile[1]}"
            chosen = np.random.default_rng(keyed_seed("slice", group)).choice(
                rows, 12 * realizations_per_group, replace=False).reshape(-1, 12)
            selected.extend(chosen)
            group_ids.extend([group] * realizations_per_group)
            if len(set(group_ids)) == max_groups:
                break
        if len(set(group_ids)) < 10:
            raise ValueError("Too few independent spatial groups in the requested slice")
        user_ids = np.asarray(selected)
        indices = user_ids.ravel()
        arrays = {"power": powers[indices]}
        del powers
        for name in ("phase", "aod_el", "aod_az", "delay"):
            arrays[name] = read_array(name)[indices]
        vectors = synthesize_ula(arrays["power"], arrays["phase"], arrays["aod_el"],
                                 arrays["aod_az"], arrays["delay"])
    channels = vectors.reshape(-1, 12, 8).transpose(0, 2, 1)
    if np.any(np.linalg.norm(channels, axis=1) == 0):
        raise ValueError("Selected slice contains a zero-energy user")
    buffer = BytesIO()
    np.savez_compressed(buffer, H=channels, group_ids=np.asarray(group_ids), user_ids=user_ids,
                        user_positions=positions[user_ids])
    if buffer.tell() > PROCESSED_CAP:
        raise ValueError("Processed slice exceeds the 20 MB cap")
    manifest = {"source_url": "https://deepmimo.net/scenarios/asu_campus_3p5",
                "scenario": "asu_campus_3p5", "frequency_hz": 3_500_000_000,
                "subcarrier": 0, "total_subcarriers": 512, "bandwidth_hz": 10_000_000,
                "bs_antennas": 8, "bs_array_shape": [8, 1], "spacing_wavelengths": .5,
                "ue_antennas": 1, "users_per_realization": 12, "realizations": len(channels),
                "spatial_groups": len(set(group_ids)),
                "spatial_grouping": f"{tile_metres} m tiles, {guard_metres} m boundary guard; "
                                    "disjoint users; adjacent retained positions stay in one group",
                "spatial_split_seed": 7, "guard_metres": guard_metres,
                "source_kind": "ray tracing, not measured over-the-air signals",
                "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "download_bytes": archive_path.stat().st_size, "download_cap_bytes": DOWNLOAD_CAP,
                "processed_bytes": buffer.tell(), "processed_cap_bytes": PROCESSED_CAP,
                "synthesis": "sum_p sqrt(10^(power_dBW/10)/512) exp(j phase) "
                             "exp(j*pi*antenna_index*sin(polar)*sin(azimuth)); reciprocal transpose",
                "generator_reference": "https://github.com/DeepMIMO/DeepMIMO/blob/"
                                       "82be994e53a2e37d88a3575a3f6fd35bb6611b53/deepmimo/generator/channel.py",
                "geometry_reference": "https://github.com/DeepMIMO/DeepMIMO/blob/"
                                      "82be994e53a2e37d88a3575a3f6fd35bb6611b53/deepmimo/generator/geometry.py"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="asu_campus_3p5")
    parser.add_argument("--output", type=Path, default=Path("data/deepmimo/asu_campus_3p5.zip"))
    parser.add_argument("--convert", type=Path, help="Convert a local archive instead of downloading")
    parser.add_argument("--slice-output", type=Path, default=Path("data/deepmimo/asu_slice.npz"))
    args = parser.parse_args()
    result = (convert_slice(args.convert, args.slice_output) if args.convert else
              acquire(args.scenario, args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
