import xarray as xr
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Union

class CCDB:
    def __init__(self, ccdb_dir: str):
        self.dir = Path(ccdb_dir)
        self._parameters = {}
        xml_path = self.dir / "ccdb.xml"
        if xml_path.exists():
            tree = ET.parse(xml_path)
            for param in tree.getroot().findall("parameter"):
                self._parameters[param.attrib["name"]] = param

    def get(self, name: str) -> Union[int, float, str, xr.DataArray]:
        param = self._parameters.get(name)
        if param is None:
            raise KeyError(f"Parameter '{name}' not found in ccdb.xml")
            
        if param.attrib.get("type") == "FILE":
            with xr.open_dataset(self.dir / f"{param.text}.nc") as ds:
                if not ds.data_vars:
                    raise ValueError(f"CCDB file for '{name}' contains no data variables.")
                # Load before closing the file so callers do not retain open handles.
                return ds[list(ds.data_vars)[0]].load()

        value = param.text
        parameter_type = param.attrib.get("type", "STRING").upper()
        if parameter_type in ("INTEGER", "INT"):
            return int(value)
        if parameter_type in ("FLOAT", "DOUBLE"):
            return float(value)
        if parameter_type in ("BOOLEAN", "BOOL"):
            normalized = value.strip().lower()
            if normalized not in ("true", "false", "1", "0"):
                raise ValueError(f"Invalid boolean value for CCDB parameter '{name}': {value}")
            return normalized in ("true", "1")
        return value

    def keys(self):
        """Names of the calibration parameters declared in ccdb.xml."""
        return tuple(self._parameters)
