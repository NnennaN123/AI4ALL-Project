# Feature Definitions for Wind Turbine Datasets

This document provides definitions for all features present in the wind turbine datasets used in this project.

## Dataset Overview

The project uses four main datasets:
1. **wtk_site_metadata.csv** - NREL Wind Toolkit site metadata
2. **uswtdb_V8_1_20250522.csv** - US Wind Turbine Database (USWTDB) version 8.1
3. **uswt_nrel_joined_match.csv** - Joined dataset matching USWTDB turbines to NREL grid cells
4. **nrel_training_data.csv** - Training dataset with turbine presence labels

---

## NREL Wind Toolkit (WTK) Features

### Location & Geography
- **site_id**: Unique identifier for each NREL grid cell/site location
- **longitude**: Longitude coordinate of the grid cell center (decimal degrees, WGS84)
- **latitude**: Latitude coordinate of the grid cell center (decimal degrees, WGS84)
- **State**: US state name where the grid cell is located
- **County**: County name where the grid cell is located

### Wind Resource & Capacity
- **wind_speed**: Average wind speed at the site (typically in m/s), representing the mean wind speed at hub height over the time period
- **capacity**: Potential wind power capacity of the site (typically in MW - megawatts), representing the maximum power output that could be generated
- **capacity_factor**: Expected capacity factor (dimensionless, typically 0-1), representing the ratio of actual energy output to maximum possible output. A capacity factor of 0.35 means the turbine produces 35% of its maximum possible output on average
- **fraction_of_usable_area**: Fraction of the grid cell area that is usable for wind development (0-1), accounting for land use restrictions, terrain, and other constraints

### Technical Data
- **power_curve**: Reference to or identifier for the power curve model used for this site, which describes how power output varies with wind speed
- **full_timeseries_directory**: Directory path containing time series wind data files for this site
- **full_timeseries_path**: Full file path to the time series wind data for this site

---

## US Wind Turbine Database (USWTDB) Features

### Identifiers
- **case_id**: Unique case identifier for each turbine record in the USWTDB
- **faa_ors**: FAA Obstruction Evaluation/Airport Airspace Analysis (OE/AAA) number, used for aviation safety evaluation
- **faa_asn**: FAA Aeronautical Study Number, another aviation-related identifier
- **usgs_pr_id**: USGS Public Release ID, unique identifier assigned by USGS for public data releases
- **eia_id**: Energy Information Administration (EIA) plant ID, linking to EIA power plant data

### Location
- **t_state**: State where the turbine is located (two-letter abbreviation)
- **t_county**: County where the turbine is located
- **t_fips**: Federal Information Processing Standards (FIPS) code for the county (5-digit code)
- **xlong**: Longitude coordinate of the turbine location (decimal degrees)
- **ylat**: Latitude coordinate of the turbine location (decimal degrees)

### Project Information
- **p_name**: Project name (wind farm name)
- **p_year**: Year the project became operational
- **p_tnum**: Number of turbines in the project
- **p_cap**: Total project capacity (MW - megawatts), the combined rated capacity of all turbines in the project

### Turbine Specifications
- **t_manu**: Turbine manufacturer name (e.g., Vestas, GE, Siemens)
- **t_model**: Turbine model designation
- **t_cap**: Individual turbine capacity (MW - megawatts), the rated power output of a single turbine
- **t_hh**: Turbine hub height (meters), the height from ground to the center of the rotor
- **t_rd**: Turbine rotor diameter (meters), the diameter of the circle swept by the rotor blades
- **t_rsa**: Rotor swept area (square meters), the area of the circle swept by the rotor blades (π × (t_rd/2)²)
- **t_ttlh**: Total turbine height (meters), the height from ground to the tip of the blade at its highest point (hub height + rotor radius)

### Turbine Status & Configuration
- **t_retrofit**: Indicates whether the turbine has been retrofitted (upgraded or modified after initial installation)
- **t_retro_yr**: Year of retrofit (if applicable)
- **t_offshore**: Indicates whether the turbine is located offshore (typically boolean or categorical)
- **t_conf_atr**: Turbine confidence attribute, indicating the reliability/confidence level of the turbine data
- **t_conf_loc**: Location confidence, indicating the reliability/confidence level of the turbine's geographic coordinates

### Imagery & Data Sources
- **t_img_date**: Date of the satellite/aerial image used to identify the turbine
- **t_img_src**: Source of the imagery used to identify the turbine (e.g., Google Earth, DigitalGlobe, NAIP)

---

## Joined Dataset Features (uswt_nrel_joined_match.csv)

This dataset contains all features from both USWTDB and WTK datasets, plus additional matching fields:

### Matching & Spatial Join Fields
- **dist_m**: Distance in meters between the turbine location and the matched NREL grid cell center
- **match_ok**: Boolean or flag indicating whether the spatial match is valid/acceptable (typically based on distance threshold)
- **geometry**: Geometric representation of the spatial features (if using GeoPandas/geospatial libraries)
- **index_right**: Index from the right dataframe after the spatial join operation

### Duplicate/Alternative Location Fields
- **longtude**: Alternative spelling/variant of longitude (may contain slight variations)
- **latitude_left**: Latitude from the left dataframe in the join
- **latitude_right**: Latitude from the right dataframe in the join

Note: The joined dataset may contain duplicate columns with different names (e.g., `longitude` vs `xlong`, `latitude` vs `ylat`) representing the same information from different sources.

---

## Training Dataset Features (nrel_training_data.csv)

This dataset contains NREL WTK features plus a target variable:

### Features (same as WTK)
- **site_id**: Unique identifier for each NREL grid cell
- **longitude**: Longitude coordinate
- **latitude**: Latitude coordinate
- **State**: US state name
- **County**: County name
- **fraction_of_usable_area**: Fraction of usable area
- **capacity**: Potential capacity
- **wind_speed**: Average wind speed
- **capacity_factor**: Expected capacity factor
- **full_timeseries_directory**: Directory path for time series data
- **full_timeseries_path**: Full path to time series data

### Target Variable
- **isTurbine**: Binary target variable indicating whether a turbine is present at this location (1 = turbine present, 0 = no turbine). This is the label used for machine learning model training.

---

## Feature Categories Summary

### Geographic Features
- Location coordinates (longitude, latitude)
- Administrative boundaries (State, County, FIPS codes)

### Physical Characteristics
- Wind resource (wind_speed, capacity_factor)
- Turbine dimensions (hub height, rotor diameter, swept area, total height)
- Site constraints (fraction_of_usable_area)

### Capacity & Power
- Individual turbine capacity (t_cap)
- Project capacity (p_cap)
- Site potential capacity (capacity)

### Temporal Features
- Project year (p_year)
- Image date (t_img_date)
- Retrofit year (t_retro_yr)

### Identification & Metadata
- Various ID fields (case_id, site_id, faa_ors, eia_id, etc.)
- Manufacturer and model information
- Data source and confidence indicators

### Spatial Matching Features
- Distance metrics (dist_m)
- Match validation flags (match_ok)
- Geometric representations (geometry)

---

## Notes

1. **Units**: Most capacity values are in megawatts (MW), distances in meters (m), heights in meters (m), and coordinates in decimal degrees.

2. **Data Sources**: 
   - USWTDB is maintained by USGS and Lawrence Berkeley National Laboratory
   - NREL Wind Toolkit is maintained by the National Renewable Energy Laboratory

3. **Spatial Matching**: The joined dataset uses spatial proximity matching (typically within 25 km radius) to link actual turbine locations (USWTDB) with potential wind resource sites (NREL WTK).

4. **Missing Values**: Some fields may contain missing values (NaN/null) depending on data availability and collection methods.
