"""Readers and chip operations for TSG CRAS rasters"""

import struct
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import pandas as pd
from dhcomp.composite import _greedy_composite
from numpy.typing import NDArray
from simplejpeg import decode_jpeg, encode_jpeg

from ..models import (
    Cras,
    CrasHeader,
    NDArrayOrZarrArray,
    SectionInfo,
    Spectra,
    TrayInfo,
)


def read_cras(filename: Union[str, Path], backing_file: Union[str, Path, None] = None) -> Cras:
    """Read a cras file

    Args:
        filename: filename to read

    Returns:
        A cras object
    """
    section_info_format: str = "4f3i"
    tray_info_format: str = "3f2i"
    head_format: str = "20s2I8h4I2h"
    with open(filename, "rb") as file:
        # using memory mapping

        # file = mmap.mmap(fopen.fileno(), 0)

        # read the 64 byte header from the .cras file
        header_bytes = file.read(64)
        # create the header information
        header_vals = struct.unpack(head_format, header_bytes)
        header = CrasHeader(header_vals[0].decode("utf-8"), *header_vals[1:])

        # Create the chunk_offset_array
        # which determines which point of the file to enter to read the .jpg image

        file.seek(64)
        b = file.read(4 * (header.nchunks + 1))
        # fmt: off
        chunk_offset_shape = (header.nchunks + 1)
        chunk_offset_array: NDArray[np.uint64] = np.ndarray(
            chunk_offset_shape, np.uint32, b
        ).astype(np.uint64)
        # deal with +4gb cras files by using uint64
        diff_offset: NDArray[np.uint64] = np.diff(chunk_offset_array, prepend=1).astype(np.uint64)
        overflow_finder: NDArray[np.int64] = np.where(diff_offset < -1)[0].astype(np.uint64)
        # fmt: on

        if len(overflow_finder) > 1:
            raise IndexError("Chunk offset array wraps around more than once")
        if len(overflow_finder) > 0:
            # add np.int32 max to the offset array this should be ok, unless there is a case where there is
            # more than 1 overflow, in which case I expect the cras reading component to crash
            chunk_offset_array[overflow_finder[0] :] += np.int64(np.iinfo(np.uint32).max + 1)

        # we currently are reading the entire cras.bip file
        # which can cause issues due to memory allocation  well handle that case
        # with some error handling here where we quit while we are ahead
        # assume that the drive backing the file has enough space to store the files
        # assume that there is enough ram to load the cras
        array_ok: bool = True
        cras: NDArrayOrZarrArray
        if backing_file is not None:
            import zarr  # ty: ignore[unresolved-import]

            # the big file flag decompresses the jpg data into a zarr array
            # with zarr it is important to ensure that you set the chunks appropriately
            # this means that you are aligning the output chunk size to the input chunk size
            outcras: Path
            if isinstance(backing_file, str):
                outcras = Path(backing_file)
            else:
                outcras = backing_file

            cras = zarr.open_array(
                str(outcras),
                mode="w",
                shape=(header.nl, header.ns, header.nb),
                chunks=(header.chunksize, header.ns, header.nb),
                dtype=np.uint8,
            )

        else:
            try:
                cras = np.zeros((header.nl, header.ns, header.nb), dtype=np.uint8)
                array_ok = True
            except MemoryError:
                print("This file is too big to fit in memory set big_file=True to dump to disk")
                array_ok = False
                cras = np.zeros(1, dtype=np.uint8)
        # if the array fits into memory then proceed to decode the .jpgs
        # using the chunk_offset_array to correctly index to the right location
        # TODO: it might be worthwhile to modify this code to manage the case
        # when you might like to have images saved if the file is too big to fit
        # into ram

        if array_ok:
            curpos: int = 0
            nr: int
            for i in range(header.nchunks):
                total_offset = (chunk_offset_array[i] + 4 * (header.nchunks + 1) + 64).astype(int)
                chunksize_in_bytes = chunk_offset_array[i + 1] - chunk_offset_array[i]
                file.seek(total_offset)
                chunk = file.read(chunksize_in_bytes)
                img = decode_jpeg(chunk, colorspace="BGR")
                # reverse the channels
                # and flip the image upsidedown
                np_image = np.flipud(img)
                nr = np_image.shape[0]
                cras[curpos : (curpos + nr), :, :] = np_image
                curpos = curpos + nr

        # the tray info section if it exists should start after the last image
        # the section info and if there is a tray info section then it should be after the tray info section
        info_table_start = (
            64 + (header.nchunks + 1) * 4 + chunk_offset_array[header.nchunks] - chunk_offset_array[0]
        ).astype(np.uint64)
        file.seek(info_table_start)

        tray: list[TrayInfo] = []
        for _i in range(header.ntrays):
            bytes = file.read(20)
            tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

        section: list[SectionInfo] = []
        for _i in range(header.nsections):
            bytes = file.read(28)
            section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))
    if backing_file is not None:
        backing_path = Path(backing_file)
        pd.DataFrame(section).to_csv(backing_path.with_suffix(".section"))
        pd.DataFrame(tray).to_csv(backing_path.with_suffix(".tray"))

    output = Cras(cras, tray, section)
    return output


def composite_spectra(spectra: Spectra, length: int = 4) -> Spectra:
    """
    Composites the spectra to a target interval length in mm
    uses dhcomp's greedy_composite internally
    keeps all the information pertaining to depth registration but looses
    all the other parameters in the scalars so there is none of the extra information that
    tsg normally provides

    """
    secdist = spectra.sampleheaders["X"].astype(float).map(np.round).copy()
    sections = spectra.sampleheaders["T"] + spectra.sampleheaders["L"]
    sections = sections.map({n: i for i, n in enumerate(sections.unique())})
    new_intervals = sections.values * 0
    int_intervals = sections.values * 0

    int_interval = 0
    for i in sections.astype(int).unique():
        sidx = sections == i
        section = sections[sidx]
        cut_points = _greedy_composite(secdist[sidx].values, length)
        tmp_section = section.values * 0
        for n, (fr, to) in enumerate(zip(cut_points[:-1], cut_points[1:], strict=True)):
            tmpidx = (secdist[sidx].values >= fr) & (secdist[sidx].values <= to)
            tmp_section[tmpidx] = n
        new_intervals[sidx.values] = tmp_section
        int_intervals[sidx.values] = tmp_section + int_interval
        int_interval += tmp_section.max() + 1

    comp_spectra = np.ones((int_interval, spectra.spectra.shape[1]))
    for i in range(int_interval):
        idx = int_intervals == i
        comp_spectra[i, :] = spectra.spectra[idx].mean(0)
    spectra.spectra = comp_spectra

    spectra.sampleheaders["new_intervals"] = new_intervals
    spectra.sampleheaders["Section"] = sections
    spectra.sampleheaders[["new_intervals"]] + spectra.sampleheaders[["Section"]]
    new_depths = (
        spectra.sampleheaders[["Section", "sample", "D", "T", "P", "X", "L", "new_intervals"]]
        .astype(float)
        .groupby(["Section", "new_intervals"])
        .max()
        .reset_index()
    )
    spectra.scalars = spectra.scalars.copy()
    spectra.scalars["new_intervals"] = new_intervals
    spectra.scalars["Section"] = sections
    new_scalay = (
        spectra.scalars[["Section", "new_intervals", "SecDist (mm)"]]
        .astype(float)
        .groupby(["Section", "new_intervals"])
        .max()
        .reset_index()
    )
    spectra.sampleheaders = new_depths
    spectra.scalars = new_scalay
    return spectra


def extract_chips(
    filename: Union[str, Path],
    outfolder: Union[str, Path],
    spectra: Spectra,
    centre_cut: bool = True,
):
    if isinstance(outfolder, str):
        outfolder = Path(outfolder)

    if not outfolder.exists():
        outfolder.mkdir()

    section_info_format: str = "4f3i"
    tray_info_format: str = "3f2i"
    head_format: str = "20s2I8h4I2h"
    file = open(filename, "rb")
    # read the 64 byte header from the .cras file
    header_bytes = file.read(64)
    # create the header information
    header_vals = struct.unpack(head_format, header_bytes)
    header = CrasHeader(header_vals[0].decode("utf-8"), *header_vals[1:])

    # Create the chunk_offset_array
    # which determines which point of the file to enter to read the .jpg image

    file.seek(64)
    b = file.read(4 * (header.nchunks + 1))
    chunk_offset_array = np.ndarray((header.nchunks + 1), np.uint32, b)

    # check for the existance of sections or  trays
    # if they exist we are going to skip ahead and import them first
    # as we are going to use them to section the images to a per spectrum basis
    # so the cras file uses compressed jpg chunks of approximately fixed dimension
    # so we need to use the section and tray information to calculate the correct
    # image size that matches the spectra
    # we will set up an array that we will use to accumulate the images into
    # then as each image is accumulated we dump it to disk and call name it the sample name
    # it is likely that we need two accumulation arrays the first as a bin to hold the images
    # as they are read to disk and the second to hold the image that we are going to export
    if header.nsections > 0 or header.ntrays > 0:
        # the tray info section if it exists should start after the last image
        # the section info and if there is a tray info section then it should be after the tray info section
        info_table_start = (
            64 + (header.nchunks + 1) * 4 + chunk_offset_array[header.nchunks] - chunk_offset_array[0]
        )
        file.seek(info_table_start)

        tray: "list[TrayInfo]" = []
        for _i in range(header.ntrays):
            bytes = file.read(20)
            tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

        section: "list[SectionInfo]" = []
        for _i in range(header.nsections):
            bytes = file.read(28)
            section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))

    # it seems to be best to allocate memory for each of the sections if there are multiple sections we
    # empty the array and create a new one of the correct dimension
    # on third thoughts we will precalculate which chunks are going to which section because we know that
    # then loop over sets of chunks dumping to disk incrementally.
    # at this stage I'm not sure it will work on drill core
    # no the header contains the chunk dimensions
    # loop over the section
    # it seems that you need to have the sample header information from the
    # nir/tir spectra we use nir because it should always be there
    # once we have that information we are going to caculate the number of pixels required
    # in the y direction that represent a single spectrum and the option will also be to dump
    # all the spectra to disk named as H_SAMPLE in a subfolder which will take an impressive amount of space
    # but such are the vagaries of ML
    # I totally assume that these headers always exist in the scalars

    # here we will do some trickery to reindex unique combinations of tray and line
    # by dropping the duplicates combinations of T and L and then reindexing
    # merging that index to the original file and then iterating over that.

    # test to see if the scalars have either ['T', 'L'] which should indicate a diamond drill hole
    # if ['Tray', 'Section'] exist then it is likely that we are dealing with chip data
    headers: list[str]
    if spectra.sampleheaders.columns.isin(["Tray", "Section"]).sum() == 2:
        headers = ["Tray", "Section"]
    elif spectra.sampleheaders.columns.isin(["T", "L"]).sum() == 2:
        headers = ["T", "L"]
    elif spectra.sampleheaders.columns.isin(["sample"]).sum() == 1:
        headers = ["sample"]
    tmp_headers: pd.DataFrame = spectra.sampleheaders[headers].drop_duplicates().reset_index()
    # get the index and set it's value to the column called index
    tmp_headers["index"] = tmp_headers.index
    # extract the index and use that as the section array
    section_array: NDArray = spectra.sampleheaders.merge(tmp_headers)["index"].values
    sample_length = spectra.scalars["SecDist (mm)"].diff()
    # this is only na for the first sample
    idx_sample_na = (sample_length.isna()) | (sample_length < 0)
    sample_length[idx_sample_na] = spectra.scalars["SecDist (mm)"][idx_sample_na]
    # pd is slow for lots of accesses

    # aggregate the spectra to modulo

    sample_array: NDArray = sample_length.values
    curchunk: int = 0
    cursample: int = 0
    processed_lines: int = 0
    yres: float
    working: NDArray[np.uint8]
    curpos: int = 0
    nr: int
    pos_fill: NDArray[np.int32]
    idx_bin_fill: NDArray[np.bool_]
    leading_bin: NDArray[np.uint8] = np.zeros((header.chunksize, header.ns, header.nb), dtype="uint8")
    total_offset: int
    chunksize_in_bytes: int
    np_image: NDArray[np.uint8]
    end_pos: int
    nextra: int
    end_np: int
    idx_section: NDArray[np.bool_]
    cut_array: NDArray[np.int32]
    n_cuts: int
    tmp_file: str
    for i, sec in enumerate(section):
        # pixel resolution
        yres = sec.utlengthmm / sec.nlines
        # allocate the section
        working = np.zeros((sec.nlines, header.ns, header.nb), dtype="uint8")
        # if we've dropped any information into the leading bin
        # dump it out into for into the working array
        if np.any(leading_bin):
            idx_bin_fill = np.all(np.any(leading_bin, 1), 1)
            pos_fill = np.where(idx_bin_fill)[0]
            working[pos_fill] = leading_bin[pos_fill]
            curpos = pos_fill[-1] + 1
            # empty the leading bin
            leading_bin = np.zeros((header.chunksize, header.ns, header.nb), dtype="uint8")
        # you need to monitor the processed lines to maintain this loop

        while (curchunk * header.chunksize - processed_lines) < sec.nlines:
            total_offset = chunk_offset_array[curchunk] + 4 * (header.nchunks + 1) + 64
            chunksize_in_bytes = chunk_offset_array[curchunk + 1] - chunk_offset_array[curchunk]
            file.seek(total_offset)
            chunk = file.read(chunksize_in_bytes)
            np_image = decode_jpeg(chunk, colorspace="BGR")
            np_image = np.flipud(np_image)
            nr = np_image.shape[0]
            end_pos = curpos + nr

            if end_pos < sec.nlines:
                working[curpos:end_pos, :, :] = np_image
            elif end_pos > sec.nlines:
                nextra = end_pos - sec.nlines
                end_pos = sec.nlines
                end_np = nr - nextra
                if nextra == 0:
                    end_np = nr
                # case when curpos is gt manage this by appending all data in this chunk to the leading bin
                if curpos < end_pos:
                    working[curpos:end_pos, :, :] = np_image[0:end_np]
                    # put the remaining information into leading bin
                    leading_bin[0:nextra, :, :] = np_image[end_np:nr]
                else:
                    # put the remaining information into leading bin
                    leading_bin[0:nr, :, :] = np_image

            elif end_pos == sec.nlines:
                # put the remaining information into leading bin
                leading_bin[0:nr, :, :] = np_image

            curpos = curpos + nr
            # increment the chunk
            curchunk += 1

        # book keeping the processed lines
        processed_lines += sec.nlines
        idx_section = section_array == i
        im_cuts = np.floor(sample_array[idx_section] / yres).astype(int)
        cut_array = np.concatenate([[0], np.cumsum(im_cuts).ravel()])
        n_cuts = len(cut_array)
        for j in range(n_cuts - 1):
            if cut_array[j + 1] <= sec.nlines:
                current_image = working[cut_array[j] : cut_array[j + 1]]
                if centre_cut:
                    # cut the image square where y i.e. depth is equal to scan width
                    mid_point = int(current_image.shape[1] / 2)
                    n_half = current_image.shape[0] // 2
                    current_image = current_image[:, (mid_point - n_half) : (mid_point + n_half), :].copy()

                tmp_file = "{}.jpg".format(cursample)
                outfile = outfolder.joinpath(tmp_file)
                outjpg = encode_jpeg(current_image)
                with open(outfile, "wb") as tmpf:
                    tmpf.write(outjpg)
                cursample += 1


def generate_chips(
    filename: Union[str, Path],
    spectra: Spectra,
    centre_cut: bool = True,
    batch_size: int = 256,
) -> Iterator[tuple[NDArray[np.uint8], ...]]:
    """
    creates an generator that generates the image tiles the last batch is not guaranteed to be the target size
    avoids having to write to folder, useful for processing files without having to first write to disk.
    """

    section_info_format: str = "4f3i"
    tray_info_format: str = "3f2i"
    head_format: str = "20s2I8h4I2h"
    with open(filename, "rb") as file:
        # read the 64 byte header from the .cras file
        header_bytes = file.read(64)
        # create the header information
        header_vals = struct.unpack(head_format, header_bytes)
        header = CrasHeader(header_vals[0].decode("utf-8"), *header_vals[1:])

        # Create the chunk_offset_array
        # which determines which point of the file to enter to read the .jpg image

        file.seek(64)
        b = file.read(4 * (header.nchunks + 1))
        chunk_offset_array = np.ndarray((header.nchunks + 1), np.uint32, b)

        # check for the existance of sections or  trays
        # if they exist we are going to skip ahead and import them first
        # as we are going to use them to section the images to a per spectrum basis
        # so the cras file uses compressed jpg chunks of approximately fixed dimension
        # so we need to use the section and tray information to calculate the correct
        # image size that matches the spectra
        # we will set up an array that we will use to accumulate the images into
        # then as each image is accumulated we dump it to disk and call name it the sample name
        # it is likely that we need two accumulation arrays the first as a bin to hold the images
        # as they are read to disk and the second to hold the image that we are going to export
        if header.nsections > 0 or header.ntrays > 0:
            # the tray info section if it exists should start after the last image
            # the section info and if there is a tray info section then it should be after the tray info
            # section
            info_table_start = (
                64 + (header.nchunks + 1) * 4 + chunk_offset_array[header.nchunks] - chunk_offset_array[0]
            )
            file.seek(info_table_start)

            tray: "list[TrayInfo]" = []
            for _i in range(header.ntrays):
                bytes = file.read(20)
                tray.append(TrayInfo(*struct.unpack(tray_info_format, bytes)))

            section: "list[SectionInfo]" = []
            for _i in range(header.nsections):
                bytes = file.read(28)
                section.append(SectionInfo(*struct.unpack(section_info_format, bytes)))

        # it seems to be best to allocate memory for each of the sections if there are multiple sections we
        # empty the array and create a new one of the correct dimension
        # on third thoughts we will precalculate which chunks are going to which section because we know that
        # then loop over sets of chunks dumping to disk incrementally.
        # at this stage I'm not sure it will work on drill core
        # no the header contains the chunk dimensions
        # loop over the section
        # it seems that you need to have the sample header information from the
        # nir/tir/mir spectra we use nir because it should always be there
        # once we have that information we are going to caculate the number of pixels required
        # in the y direction that represent a single spectrum and the option will also be to dump
        # all the spectra to disk named as H_SAMPLE in a subfolder which will take an impressive amount of
        # space but such are the vagaries of ML
        # I totally assume that these headers always exist in the scalars

        # here we will do some trickery to reindex unique combinations of tray and line
        # by dropping the duplicates combinations of T and L and then reindexing
        # merging that index to the original file and then iterating over that.

        # test to see if the scalars have either ['T', 'L'] which should indicate a diamond drill hole
        # if ['Tray', 'Section'] exist then it is likely that we are dealing with chip data
        headers: list[str]
        if spectra.sampleheaders.columns.isin(["Tray", "Section"]).sum() == 2:
            headers = ["Tray", "Section"]
        elif spectra.sampleheaders.columns.isin(["T", "L"]).sum() == 2:
            headers = ["T", "L"]
        elif spectra.sampleheaders.columns.isin(["sample"]).sum() == 1:
            headers = ["sample"]
        tmp_headers: pd.DataFrame = spectra.sampleheaders[headers].drop_duplicates().reset_index()
        # get the index and set it's value to the column called index
        tmp_headers["index"] = tmp_headers.index
        # extract the index and use that as the section array
        section_array: NDArray = spectra.sampleheaders.merge(tmp_headers)["index"].values
        sample_length = spectra.scalars["SecDist (mm)"].diff()
        # this is only na for the first sample
        idx_sample_na = (sample_length.isna()) | (sample_length < 0)
        sample_length[idx_sample_na] = spectra.scalars["SecDist (mm)"][idx_sample_na]
        # pd is slow for lots of accesses

        # aggregate the spectra to modulo

        sample_array: NDArray = sample_length.values
        curchunk: int = 0
        cursample: int = 0
        processed_lines: int = 0
        yres: float
        working: NDArray[np.uint8]
        curpos: int = 0
        nr: int
        pos_fill: NDArray[np.int32]
        idx_bin_fill: NDArray[np.bool_]
        leading_bin: NDArray[np.uint8] = np.zeros((header.chunksize, header.ns, header.nb), dtype="uint8")
        total_offset: int
        chunksize_in_bytes: int
        np_image: NDArray[np.uint8]
        end_pos: int
        nextra: int
        end_np: int
        idx_section: NDArray[np.bool_]
        cut_array: NDArray[np.int32]
        n_cuts: int
        cutouts: list[NDArray[np.uint8]] = []
        biter: int = 0
        for i, sec in enumerate(section):
            # pixel resolution
            yres = sec.utlengthmm / sec.nlines
            # allocate the section
            working = np.zeros((sec.nlines, header.ns, header.nb), dtype="uint8")
            # if we've dropped any information into the leading bin
            # dump it out into for into the working array
            if np.any(leading_bin):
                idx_bin_fill = np.all(np.any(leading_bin, 1), 1)
                pos_fill = np.where(idx_bin_fill)[0]
                working[pos_fill] = leading_bin[pos_fill]
                curpos = pos_fill[-1] + 1
                # empty the leading bin
                leading_bin = np.zeros((header.chunksize, header.ns, header.nb), dtype="uint8")
            # you need to monitor the processed lines to maintain this loop

            while (curchunk * header.chunksize - processed_lines) < sec.nlines:
                total_offset = chunk_offset_array[curchunk] + 4 * (header.nchunks + 1) + 64
                chunksize_in_bytes = chunk_offset_array[curchunk + 1] - chunk_offset_array[curchunk]
                file.seek(total_offset)
                chunk = file.read(chunksize_in_bytes)
                np_image = decode_jpeg(chunk, colorspace="BGR")
                np_image = np.flipud(np_image)
                nr = np_image.shape[0]
                end_pos = curpos + nr

                if end_pos < sec.nlines:
                    working[curpos:end_pos, :, :] = np_image
                elif end_pos > sec.nlines:
                    nextra = end_pos - sec.nlines
                    end_pos = sec.nlines
                    end_np = nr - nextra
                    if nextra == 0:
                        end_np = nr
                    # case when curpos is gt manage this by appending all data in this chunk
                    # to the leading bin
                    if curpos < end_pos:
                        working[curpos:end_pos, :, :] = np_image[0:end_np]
                        # put the remaining information into leading bin
                        leading_bin[0:nextra, :, :] = np_image[end_np:nr]
                    else:
                        # put the remaining information into leading bin
                        leading_bin[0:nr, :, :] = np_image

                elif end_pos == sec.nlines:
                    # put the remaining information into leading bin
                    leading_bin[0:nr, :, :] = np_image

                curpos = curpos + nr
                # increment the chunk
                curchunk += 1

            # book keeping the processed lines
            processed_lines += sec.nlines
            idx_section = section_array == i
            im_cuts = np.floor(sample_array[idx_section] / yres).astype(int)
            cut_array = np.concatenate([[0], np.cumsum(im_cuts).ravel()])
            n_cuts = len(cut_array)

            for j in range(n_cuts - 1):
                if cut_array[j + 1] <= sec.nlines:
                    current_image = working[cut_array[j] : cut_array[j + 1]]
                    if centre_cut:
                        # cut the image square where y i.e. depth is equal to scan width
                        mid_point = int(current_image.shape[1] / 2)
                        n_half = current_image.shape[0] // 2
                        current_image = current_image[
                            :, (mid_point - n_half) : (mid_point + n_half), :
                        ].copy()
                    cutouts.append(current_image)
                    cursample += 1

                # if (biter >= batch_size) or (i >=header.nsections) or (len(cutouts)>=batch_size):
                if (cursample % batch_size == 0) or (i >= header.nsections):
                    biter = 0
                    tupout = tuple(cutouts)
                    cutouts = []
                    yield tupout
                else:
                    biter += 1
        yield tuple(cutouts)


__all__ = ["read_cras", "composite_spectra", "extract_chips", "generate_chips"]
