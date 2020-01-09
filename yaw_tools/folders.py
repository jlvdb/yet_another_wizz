import os
import re
import shutil
from collections import OrderedDict


DEFAULT_EXT_DATA = ".dat"
DEFAULT_EXT_BOOT = ".boot"


def getext(path):
    try:
        base, ext = os.path.splitext(path)
        return ext.lstrip(".")
    except IndexError:
        return ""


def binname(path):
    pattern = "\d\.\d*z\d\.\d*"
    try:
        name = re.findall(pattern, path)[-1]
    except IndexError:
        raise ValueError("could not identify bin naming pattern")
    return name


class Folder(object):

    def __init__(self, rootpath, wipe=False):
        self.root = os.path.expanduser(rootpath)
        if wipe:
            shutil.rmtree(self.root)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def copy_to(self, path):
        if path == self.root:
            raise OSError("source and destination are the same")
        shutil.copytree(self.root, os.path.expanduser(path))
        return CCFolder(path)

    def join(self, *args):
        return os.path.join(self.root, *args)

    def mkdir(self, subdirname, wipe=False):
        path = self.join(subdirname)
        if wipe and os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        return path

    def listdir(self):
        return [self.join(f) for f in os.listdir(self.root)]

    def zbin_filename(self, zmin, zmax, ext, prefix=None, suffix=None):
        parts = ["%.3fz%.3f" % (zmin, zmax)]
        if prefix is not None:
            parts.insert(0, prefix)
        if suffix is not None:
            parts.append(suffix)
        fname = "_".join(parts)
        fname = ".".join([fname, ext.lstrip(".")])
        return self.join(fname)

    def incorporate(self, path, ext=None):
        root, fname = os.path.split(path)
        if ext is not None:
            new_ext = ext.lstrip(".")
            core, old_ext = os.path.splitext(fname)
            fname = ".".join([core, new_ext])
        return self.join(fname)

    def find(self, pattern):
        matches = []
        search_pattern = re.compile(pattern)
        for root, dirs, files in os.walk(self.root):
            for items in (dirs, files):
                for item in items:
                    if search_pattern.match(item):
                        matches.append(os.path.join(root, item))
        return matches


class ScaleFolder(Folder):

    def path_autocorr_file(self, ext, suffix=None):
        if suffix is None:
            fname = "autocorr.%s" % ext.lstrip(".")
        else:
            fname = "autocorr_%s.%s" % (suffix, ext.lstrip("."))
        return self.join(fname)

    def path_crosscorr_file(self, ext, zlims=None):
        if zlims is None:
            fname = self.join("crosscorr.%s" % ext.lstrip("."))
        elif type(zlims) is str:
            fname = self.join("crosscorr_%s.%s" % (zlims, ext.lstrip(".")))
        else:
            zmin, zmax = zlims
            fname = self.zbin_filename(zmin, zmax, ext, prefix="crosscorr")
        return self.join(fname)

    def path_weights_file(self):
        return self.join("bin_weights.pickle")

    def list_autocorr_files(self, ext):
        suffixes = OrderedDict()
        for file in os.listdir(self.root):
            # check if the file name has the correct prefix
            if not file.startswith("autocorr"):
                continue
            # check if the type of file is correct
            try:
                if getext(file) == ext.lstrip("."):
                    # extract the information
                    base, file_ext = os.path.splitext(file)
                    name, suffix = base.split("_")
                    suffixes[suffix] = self.join(file)
            except KeyError:
                continue  # unknown file type
            except IndexError:
                suffixes[None] = self.join(file)
        return suffixes

    def list_crosscorr_files(self, ext):
        zbins = OrderedDict()
        for file in os.listdir(self.root):
            # check if the file name has the correct prefix
            if not file.startswith("crosscorr"):
                continue
            try:
                # check if the type of file is correct
                if getext(file) == ext.lstrip("."):
                    # extract the information
                    base, file_ext = os.path.splitext(file)
                    name, zbin = base.split("_")
                    zbins[zbin] = self.join(file)
            except KeyError:
                continue  # unknown file type
            except IndexError:
                zbins[None] = self.join(file)
        return zbins


class CCFolder(Folder):

    _pattern_scale = re.compile("kpc\d*t\d*")

    def __init__(self, rootpath, wipe=False):
        super().__init__(rootpath, wipe)
        # find existing scales
        self.scales = OrderedDict()
        for scale_name in os.listdir(self.root):
            if self._pattern_scale.fullmatch(scale_name):
                self.scales[scale_name] = ScaleFolder(self.join(scale_name))

    def __getitem__(self, key):
        if key not in self.list_scalenames():
            raise KeyError(
                "scale '%s' does not exist in '%s'" % (key, self.root))
        else:
            return self.scales[key]

    def __contains__(self, scale):
        return scale in self.scales

    def path_binning_file(self):
        return self.join("binning%s" % DEFAULT_EXT_DATA)

    def path_params_file(self):
        return self.join("yet_another_wizz.param")

    def add_scale(self, rlims):
        if type(rlims) is str:
            scale_name = rlims
        else:
            rmin, rmax = rlims
            scale_name = "kpc%dt%d" % (rmin, rmax)
        if scale_name not in self.scales:
            self.scales[scale_name] = ScaleFolder(self.join(scale_name))
        return scale_name

    def pop_scale(self, scale):
        return self.scales.pop(scale)

    def list_scalenames(self):
        return tuple(self.scales.keys())

    def list_scalepaths(self):
        return tuple(scale.root for scale in self.scales.values())

    def iter_scales(self):
        return self.scales.items()

    def find_autocorr_files(self, ext):
        return self.find("autocorr.*%s" % ext.lstrip("."))

    def find_crosscorr_files(self, ext):
        return self.find("crosscorr.*%s" % ext.lstrip("."))

    def copy_meta_to(self, path):
        other = CCFolder(path)
        # copy meta files
        for method in ("path_binning_file", "path_params_file"):
            src = getattr(self, method)()
            dst = getattr(other, method)()
            if os.path.exists(src):
                shutil.copy(src, dst)
        # register the scales and copy the weights
        for scale in self.scales:
            other.add_scale(scale)
            src = self.scales[scale].path_weights_file()
            dst = other.scales[scale].path_weights_file()
            if os.path.exists(src):
                shutil.copy(src, dst)
        return other


if __name__ == "__main__":
    from pprint import pprint

    ff = Folder("~/CC/YAW/MICE2_KV450_magnification_on/n_cc/idealized")
    print("# redshift bin file names")
    pprint(
        ff.zbin_filename(0.101, 0.301, ".txt", prefix="test", suffix="stuff"))
    pprint(ff.zbin_filename(0.101, 0.301, ".txt", prefix="test"))
    pprint(ff.zbin_filename(0.101, 0.301, ".txt", suffix="stuff"))
    pprint(ff.zbin_filename(0.101, 0.301, ".txt"))
    print("# incorporation")
    old_path = "testdir/test_0.101z0.301.txt"
    print("old path: %s" % old_path)
    new_path = ff.incorporate(old_path, ext=".pickle")
    print("new path: %s" % new_path)
    print("# listing and finding files")
    pprint(ff.listdir())
    pprint(ff.find(".*0\.301z0\.501.*"))

    print()

    sf = ScaleFolder(
        "~/CC/YAW/MICE2_KV450_magnification_on/n_cc/idealized/kpc100t1000")
    print("# file path proposals")
    pprint(sf.path_autocorr_file(".pickle"))
    pprint(sf.path_autocorr_file(".pickle", suffix="spec"))
    pprint(sf.path_crosscorr_file(DEFAULT_EXT_DATA))
    pprint(sf.path_crosscorr_file(DEFAULT_EXT_DATA, [0.101, 0.301]))
    pprint(sf.path_weights_file())
    print("# listing files")
    pprint(sf.list_autocorr_files(DEFAULT_EXT_DATA))
    pprint(sf.list_autocorr_files(DEFAULT_EXT_DATA))
    pprint(sf.list_crosscorr_files(DEFAULT_EXT_DATA))
    pprint(sf.list_crosscorr_files(DEFAULT_EXT_DATA))

    print()

    cf = CCFolder("~/CC/YAW/MICE2_KV450_magnification_on/n_cc/idealized")
    print("# file path proposals")
    pprint(cf.path_binning_file())
    pprint(cf.path_params_file())
    print("# listing scales")
    pprint(cf.list_scalenames())
    pprint(cf.list_scalepaths())
    pprint(tuple(cf.iter_scales()))
    print("# finding files")
    pprint(cf.find_autocorr_files(DEFAULT_EXT_DATA))
    pprint(cf.find_autocorr_files(DEFAULT_EXT_DATA))
    pprint(cf.find_crosscorr_files(DEFAULT_EXT_DATA))
    print("# keyword lookup and membership tests")
    pprint(cf["kpc100t1000"].root)
    pprint("kpc100t1000" in cf)
    pprint(cf.pop_scale("kpc100t1000").root)
    pprint(cf.list_scalenames())

    print()
