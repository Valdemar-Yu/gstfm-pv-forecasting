# DKASC input data

Obtain measurements from the [DKA Solar Centre](https://dkasolarcentre.com.au/), for Site 7 First Solar and Site 1B Trina. Measurements are not bundled. `input_manifest.json` records the exact processed input filenames, SHA-256 hashes, columns, row counts and test dates used for the archived results. Matching a hash is required for an exact-input reproduction; newly downloaded or reprocessed data may differ.

Each CSV is chronological and contains six columns in this exact order:

`date,Active_Power,Weather_Temperature_Celsius,Weather_Relative_Humidity,Global_Horizontal_Radiation,Diffuse_Horizontal_Radiation`

Power is in kW. Weather inputs are historical observations. Seasons follow the southern hemisphere: spring September–November, summer December–February, autumn March–May, winter June–August. Filenames retain the original year-span labels; exact test dates are in the manifest. Do not infer inclusion of all dates from a filename. Some source hourly records have gaps; forecast horizons count consecutive retained records in the archived protocol.

Place the eight CSV files at the paths in the manifest. Run `python scripts/check_inputs.py` before training. The archived files contain no missing cells. The loader clips negative target power and fits a StandardScaler on training rows only; validation and test windows include the preceding 24 context records. Split sizes are floor(0.8 N), the remaining validation rows, and floor(0.1 N) test rows. For H-step direct forecasts there are `test_hours - H + 1` forecast origins. Consecutive multi-step forecasts overlap and are not independent samples.

The release does not contain the original raw-to-hourly preprocessing script. Exact end-to-end replication from raw measurements therefore requires the processed inputs identified in the manifest; the included aggregate result files support verification of reported tables without those inputs.
