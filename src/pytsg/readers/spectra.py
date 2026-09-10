"""Readers for TSG spectral metadata and BIP data pairs."""

import re
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..models import BandHeaders, ClassHeaders, Spectra


def _read_tsg_file(filename: Union[str, Path]) -> list[str]:
    """Read a TSG metadata file and strip line endings."""
    lines: list[str] = []
    with open(filename, encoding="cp1252") as file:
        for line in file:
            lines.append(line.rstrip())
    return lines


def _find_header_sections(tsg_str: list[str]) -> dict[str, tuple[int, int]]:
    """Find the sections in a TSG metadata file."""
    re_strip: re.Pattern[str] = re.compile(r"^\[[a-zA-Z0-9 ]+\]")
    positions: list[int] = []
    for i, line in enumerate(tsg_str):
        if len(re_strip.findall(line)) > 0:
            positions.append(i)

    positions.append(len(tsg_str))
    sections: dict[str, tuple[int, int]] = {}
    for i in range(len(positions) - 1):
        section = (positions[i] + 1, positions[i + 1] - 1)
        name = tsg_str[positions[i]].strip("[]")
        sections.update({name: section})
    return sections


def _parse_kvp(line: str, split: str = "=") -> dict[str, str]:
    """Parse one separator-delimited line into a key/value dictionary."""
    if line.find(split) >= 0:
        split_line = line.split(split)
        key = split_line[0].strip()
        value = split_line[1].strip()
        return {key: value}
    return {}


def _parse_section(section_list: list[str], key_split: str = ":") -> list[dict[str, str]]:
    """Parse the generic key/value lines in a TSG section."""
    return [_parse_kvp(line, key_split) for line in section_list]


def _parse_sample_header(section_list: list[str], key_split: str = ":") -> list[dict[str, str]]:
    """Parse sample-header records from a TSG metadata section."""
    final: list[dict[str, str]] = []
    for line in section_list:
        key_values = _parse_kvp(line, key_split)
        key = list(key_values.keys())[0]
        sample: dict[str, str] = {"sample": key}
        for item in key_values[key].split():
            parsed = _parse_kvp(item)
            if parsed is not None:
                sample.update(parsed)
        final.append(sample)
    return final


def _parse_class_section(section_list: list[str], classnumber: int) -> ClassHeaders:
    """Parse one class definition section."""
    class_names: dict[str, str] = {}
    class_info: dict[int, str] = {}
    for line in section_list:
        if line.find("=") >= 0:
            split_line = line.split("=")
            class_names[split_line[0].strip()] = split_line[1].strip()
        elif line.find(":") >= 0:
            split_line = line.split(":")
            class_info[int(split_line[0])] = split_line[1]

    max_class = int(class_names["max"])
    colors_list: list[int] = []
    if "colours" in class_names:
        colors_list = [int(value) for value in class_names["colours"].split(" ")]

    return ClassHeaders(classnumber, class_names["name"], max_class, class_info, colors=colors_list)


def _parse_wavelength_specs(line: str) -> dict[str, Union[float, str]]:
    """Parse a TSG wavelength range line."""
    split_wavelength = line.split()
    return {
        "start": float(split_wavelength[0]),
        "end": float(split_wavelength[1]),
        "unit": split_wavelength[-1],
    }


def _read_bip(filename: Union[str, Path], coordinates: dict[str, str]) -> NDArray[np.float32]:
    """Read and reshape the two-plane TSG BIP array."""
    tmp_array: NDArray[np.float32] = np.fromfile(filename, dtype=np.float32)
    n_bands = int(coordinates["lastband"])
    n_samples = int(coordinates["lastsample"])
    return np.reshape(tmp_array, (2, n_samples, n_bands))


def _calculate_wavelengths(wavelength_specs: dict[str, float], coordinates: dict[str, str]) -> NDArray:
    """Calculate evenly spaced wavelengths from TSG metadata."""
    wavelength_range = wavelength_specs["end"] - wavelength_specs["start"]
    resolution = wavelength_range / (int(coordinates["lastband"]) - 1)
    return np.arange(wavelength_specs["start"], wavelength_specs["end"] + resolution, resolution)


def _parse_bandheaders(bandheaders: list[str]) -> list[BandHeaders]:
    """Parse scalar band header records."""
    output: list[BandHeaders] = []
    for band_header in bandheaders:
        split_header = band_header.split(":")
        band = int(split_header[0])
        info = split_header[1]
        split_info = info.split(";")
        name = split_info[0]
        if len(split_info) > 1:
            flag = int(split_info[3])
            if flag <= 2:
                class_name: Union[int, str, float] = int(split_info[4])
            elif flag == 13:
                class_name = split_info[4]
            else:
                class_name = float(split_info[4])
        else:
            class_name = -1
            flag = -1
        output.append(BandHeaders(band, name, class_name, flag))
    return output


def _parse_tsg(fstr: list[str], headers: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Parse all supported sections from a TSG metadata file."""
    d_info: dict[str, Any] = {}
    for key in headers:
        start, end = headers[key]
        if key == "sample headers":
            d_info[key] = pd.DataFrame(_parse_sample_header(fstr[start:end], ":"))
        elif key == "wavelength specs":
            d_info[key] = _parse_wavelength_specs(fstr[start:end][0])
        elif key == "band headers":
            d_info[key] = _parse_bandheaders(fstr[start:end])
        elif key.find("class") == 0:
            class_number = int(key.split(" ")[1])
            class_info = _parse_class_section(fstr[start:end], class_number)
            if "class" in d_info:
                d_info["class"].update({class_number: class_info})
            else:
                d_info["class"] = {class_number: class_info}
        else:
            parsed_section: dict[str, str] = {}
            for line in fstr[start:end]:
                parsed_section.update(_parse_kvp(line))
            d_info[key] = parsed_section
    return d_info


def _parse_scalars(
    scalars: NDArray,
    classes: dict[int, ClassHeaders],
    bandheaders: list[BandHeaders],
    nodata: int = -1,
) -> pd.DataFrame:
    """Map scalar bands to a named Pandas DataFrame."""
    tmp_series: list[pd.DataFrame] = []
    for band_header in bandheaders:
        band_value = scalars[:, band_header.band]
        if band_header.flag == 2:
            if isinstance(band_header.class_number, int) and band_header.class_number > 0:
                bv = np.where(
                    np.isclose(band_value, np.finfo("float32").min),
                    -1,
                    band_value,
                ).astype(int)
                mapped = classes[band_header.class_number].map_ints(bv)
                tmp_series.append(pd.DataFrame(mapped, columns=[band_header.name]))
            else:
                tmp_series.append(pd.DataFrame(band_value, columns=[band_header.name]))
        else:
            tmp_series.append(pd.DataFrame(band_value, columns=[band_header.name]))
    return pd.concat(tmp_series, axis=1)


def read_spectra(
    tsg_file: Union[Path, str],
    bip_file: Union[Path, str],
    *,
    spectrum_name: str = "nir",
) -> Spectra:
    """Read one TSG metadata/data spectral pair eagerly."""
    fstr = _read_tsg_file(tsg_file)
    headers = _find_header_sections(fstr)
    info = _parse_tsg(fstr, headers)
    spectra = _read_bip(bip_file, info["coordinates"])
    wavelength = _calculate_wavelengths(info["wavelength specs"], info["coordinates"])
    scalars = _parse_scalars(spectra[1, :, :], info["class"], info["band headers"])

    return Spectra(
        spectrum_name,
        spectra[0, :, :],
        wavelength,
        info["sample headers"],
        info["class"],
        info["band headers"],
        scalars,
    )


__all__ = ["read_spectra"]
