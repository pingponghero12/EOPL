from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent

# Create directories
(ROOT / "input").mkdir(exist_ok=True)
(ROOT / "ccdb").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

# 1. Generate 100x100x2 uniform image
data = np.ones((2, 100, 100), dtype=np.float32)
ds = xr.Dataset(
    {"default_image": (["Channels", "ALT", "ACT"], data)},
    coords={"Channels": [1, 2], "ALT": np.arange(100), "ACT": np.arange(100)}
)
ds.to_netcdf(ROOT / "input/input_image.nc")

# 2. Generate dummy CCDB gain array (size 50, to test interpolation to 100)
gain_data = np.linspace(0.9, 1.1, 50, dtype=np.float32)
gain_ds = xr.Dataset({"gain_array_1d": (["ACT"], gain_data)})
gain_ds.to_netcdf(ROOT / "ccdb/gain_array_1d.nc")

with open(ROOT / "ccdb/ccdb.xml", "w") as f:
    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<CalibrationDataBase version="01.00.00">
    <parameter name="gain_array_1d" description="Per-pixel gain" type="FILE" units="dimensionless">gain_array_1d</parameter>
</CalibrationDataBase>""")

# 3. Create XML configs
with open(ROOT / "global.xml", "w") as f:
    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<Global_Configuration version="01.00.00">
    <parameter name="MissionID" type="STRING">Sentinel3</parameter>
    <parameter name="CCDBPath" type="STRING">ccdb/</parameter>
    <parameter name="RandomSeed" type="INTEGER">42</parameter>
</Global_Configuration>""")

with open(ROOT / "local.xml", "w") as f:
    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<IOM_Local_Configuration version="01.00.00">
    <parameter name="Lambda" type="FLOAT" units="nm">500.3</parameter>
</IOM_Local_Configuration>""")

print("Test data and configurations generated successfully.")
