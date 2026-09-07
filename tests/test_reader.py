import tempfile
import unittest

from src.pytsg import parse_tsg


class TestFileReaders(unittest.TestCase):
    def test_read_package(self):
        folder = r"example_data/SWMB007d"
        tmp_data = parse_tsg.read_package(folder, read_cras_file=True)
        self.assertTrue(hasattr(tmp_data, "nir"))
        self.assertTrue(hasattr(tmp_data, "tir"))
        self.assertTrue(hasattr(tmp_data, "cras"))

    def test_extract_chips(self):
        folder = r"example_data/SWMB007d"
        cras_file = folder + "/SWMB007d_chips_tsg_cras.bip"
        tsg_file = folder + "/SWMB007d_chips_tsg.tsg"
        bip_file = folder + "/SWMB007d_chips_tsg.bip"

        spectra = parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, "nir")
        with tempfile.TemporaryDirectory() as tmpdirname:
            parse_tsg.extract_chips(cras_file, tmpdirname, spectra)

    def test_generate_chips(self):
        folder = r"example_data/SWMB007d"
        cras_file = folder + "/SWMB007d_chips_tsg_cras.bip"
        tsg_file = folder + "/SWMB007d_chips_tsg.tsg"
        bip_file = folder + "/SWMB007d_chips_tsg.bip"

        spectra = parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, "nir")
        chip_generator = parse_tsg.generate_chips(cras_file, spectra,batch_size=12)
        bsize:list[int] = [12, 12, 12, 12, 12, 12, 0]
        actual:list[int] = []
        for i in chip_generator:
            actual.append(len(i))
        self.assertListEqual(bsize, actual)
if __name__ == "__main__":
    unittest.main()
"""
    def test_composite_spectra(self):
        folder = r"example_data/27313_NDDH0505_Savage_River"
        cras_file = folder + "/27313_NDDH0505_Savage_River_tsg.bip"
        tsg_file = folder + "/27313_NDDH0505_Savage_River_tsg.tsg"
        bip_file = folder + "/27313_NDDH0505_Savage_River_tsg.bip"

        spectra = parse_tsg.read_tsg_bip_pair(tsg_file, bip_file, "nir")
        composite_spectra = parse_tsg.composite_spectra(spectra, length=32)
        # comp the spectra check that the 3 output channels are the same size
        outsize = [
            composite_spectra.spectra.shape[0],
            composite_spectra.scalars.shape[0],
            composite_spectra.sampleheaders.shape[0],
        ]
        self.assertListEqual([93, 93, 93], outsize)
"""
