import logging
import re
from typing import Dict, IO, Iterator, Sequence, Union

from pyteomics import mgf
from spectrum_utils.spectrum import MsmsSpectrum
from typing import IO, Union, Sequence, Iterator
import os
import lzma
import gzip
import bz2



logger = logging.getLogger('gleams')


def get_spectra(source: Union[IO, str], scan_nrs: Sequence[int] = None)\
        -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given MGF file, optionally filtering by
    scan number.

    Parameters
    ----------
    source : Union[IO, str]
        The MGF source (file name or open file object) from which the spectra
        are read.
    scan_nrs : Sequence[int]
        Only read spectra with the given scan numbers (enumerate positions). 
        If `None`, no filtering on scan number is performed.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    with mgf.MGF(source) as f_in:
        # Iterate over a subset of spectra filtered by scan number.
        if scan_nrs is not None:
            scan_set = set(scan_nrs)  # Convert to set for O(1) lookup
            found_count = 0
            def spectrum_it():
                nonlocal found_count
                for scan_nr, spectrum_dict in enumerate(f_in):
                    if scan_nr in scan_set:
                        found_count += 1
                        # Add enumerate position as identifier for matching metadata
                        spectrum_dict['_scan_position'] = scan_nr
                        yield spectrum_dict
                        # Early exit if we've found all requested scans
                        if found_count >= len(scan_nrs):
                            return

        
        # Or iterate over all MS/MS spectra.
        else:
            print("iterating over all spectra")
            def spectrum_it():
                for scan_nr, spectrum_dict in enumerate(f_in):
                    spectrum_dict['_scan_position'] = scan_nr
                    yield spectrum_dict

        for spectrum in spectrum_it():
            try:
                yield _parse_spectrum(spectrum)
            except ValueError as e:
                pass
                # logger.warning(f'Failed to read spectrum '
                #                f'{spectrum["params"]["title"]}: %s', e)




def _parse_spectrum(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.
    """
    params = spectrum_dict.get("params", {})

    # -------------------------
    # Identifier / scan number
    # -------------------------
    # Use enumerate position if available (added by get_spectra)
    # This matches the 'scan' column in metadata
    if '_scan_position' in spectrum_dict:
        identifier = spectrum_dict['_scan_position']
    else:
        # Fallback to extracting from title (old behavior)
        title = params.get("title", "")
        match = re.search(r"scan:(\d+)", title)
        if not match:
            raise ValueError(f"Could not extract scan number from title: {title}")
        identifier = int(match.group(1))

    # -------------------------
    # Fragment peaks
    # -------------------------
    mz_array = spectrum_dict["m/z array"]
    intensity_array = spectrum_dict["intensity array"]

    # -------------------------
    # Retention time
    # -------------------------
    retention_time = params.get("rtinseconds")
    if retention_time is not None:
        retention_time = float(retention_time)
    else:
        retention_time = None  # or 0.0 if your model requires a value

    # -------------------------
    # Precursor m/z
    # -------------------------
    pepmass = params.get("pepmass")
    if pepmass is None:
        raise ValueError("Missing PEPMASS in spectrum params")

    if isinstance(pepmass, (tuple, list)):
        precursor_mz = float(pepmass[0])
    else:
        precursor_mz = float(pepmass)

    # -------------------------
    # Precursor charge
    # -------------------------
    charge = params.get("charge")
    if charge is None:
        raise ValueError("Missing precursor charge")

    if isinstance(charge, (list, tuple)):
        charge = charge[0]

    if isinstance(charge, str):
        charge = charge.rstrip("+")

    precursor_charge = int(charge)
    
    #print("Done parsing spectrum")
    #print(f"Parsed spectrum: {identifier}, Precursor m/z: {precursor_mz}, "
    #      f"Precursor charge: {precursor_charge}, Retention time: {retention_time}, "
    #        f"m/z array: {mz_array}, Intensity array: {intensity_array}")

    #spectrum = MsmsSpectrum(str(identifier), precursor_mz, precursor_charge,
                            #mz_array, intensity_array, None, retention_time)
    
    spectrum = MsmsSpectrum(str(identifier), precursor_mz, precursor_charge,
                            mz_array, intensity_array, retention_time)
    
    #print("Returning spectrum")
    return spectrum
