import numpy as np
import matplotlib.pyplot as plt
import tables
import pandas as pd
import astropy
import astropy.units as u

from tables import *

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz

data = pd.read_hdf("ARCA.KM3Online.00000133_data_for_GRB221009A_SSIFIC2023.h5")

#GENERAL VISUALIZATION OF DATA (COMPACT)------------------
print("Set of all data for GRB221009A\n", data)

#VISUALIZATION OF ALL COLUMNS -----------------------------
print(data.columns)

#TIME WINDOW OF THE SAMPLE ----------------------------------
tmax = data["MJD"][data["MJD"].idxmax()]
tmin = data["MJD"][data["MJD"].idxmin()]
twindow = (tmax-tmin)*24*3600
print("\n Time window of ", twindow, " seconds, equivalent to ", twindow/3600, " hours \n")

#MINIMUM QUALITY CUTS --------------------------------------------
data_mqc = data[(data["Energy"]>10.0) & (data["RecoQuality"]>0.0) & (data["Track Length"]>0.0) & (data["RecoBeta0"]>0.0)]
print("Number of events after minimal quality cut:", len(data_mqc))

#ANALYSIS CUTS ---------------------------------------------------
data_an = data_mqc[(data_mqc["Energy"]>1000.0) & (data_mqc["RecoQuality"]>155.0)]
print("Number of events after analysis cut:", len(data_an))


# DATA ANALYSIS  -------------------------------------------------------------------------

#New column: Elevation for each event [deg]
data_an["Elevation"] = np.rad2deg(np.arccos(data_an["BestMuonDz"]))-90

#Crucial times for analysis: T0-50s, T0, T0+5000s
T0 = "2022-10-09T13:16:59.000"

T_burst = Time(T0, format="isot", scale="utc") #time object
T_window_in = T_burst - 50*u.second
T_window_out = T_burst + 5000*u.second

#Calculation of GRB coordinates in local frame (time, eq coord, detector position)

# 1. Locate detector
arca_lat = 36 + (16/60)         #deg
arca_long = 16 + (6/60)         #deg
arca_alt = -3500                #meters

telescope = EarthLocation.from_geodetic(lat=arca_lat, lon=arca_long, height=arca_alt)

#2. Introducing equatorial coordinates for GRB
grb_radec = SkyCoord("19h13m03s", "+19d48m09s", frame="icrs")

#3. Introducing parameters from local frame (location & time) at beginning & end of T_ON
obs_window_in = AltAz(location=telescope, obstime=T_window_in)
obs_window_out = AltAz(location=telescope, obstime=T_window_out)

#4. Transform to local coordinates
grb_altaz_in = grb_radec.transform_to(obs_window_in)
grb_altaz_out = grb_radec.transform_to(obs_window_out)

print("GRB elevation at the beginning of time window T_ON is: ", grb_altaz_in.alt.deg, " deg")
print("GRB elevation at the end of time window T_ON is: ", grb_altaz_out.alt.deg, " deg\n")

#Creation of analysis region in elevation band
elevation_min = grb_altaz_in.alt.deg-2      #deg
elevation_max = grb_altaz_out.alt.deg+2     #deg

data_elevation_band = data_an[(data_an["Elevation"]>elevation_min) & (data_an["Elevation"]<elevation_max)]
print("Number of events inside elevation band: ", len(data_elevation_band), "\n")

#Analysis band in T_OFF
data_elevation_band_OFF = data_elevation_band[(data_elevation_band["MJD"]<T_window_in.mjd) | 
                                              (data_elevation_band["MJD"]>T_window_out.mjd)]

N_off = len(data_elevation_band_OFF)

#Analysis band in T_ON
data_elevation_band_ON = data_elevation_band[(data_elevation_band["MJD"]>T_window_in.mjd) & 
                                              (data_elevation_band["MJD"]<T_window_out.mjd)]

N_on = len(data_elevation_band_ON)

print("Number of events inside the elevation band during T_OFF: ", N_off)
print("Number of events inside the elevation band during T_ON: ", N_on)

#Time intervals T_OFF & T_ON
T_on = (T_window_out.mjd - T_window_in.mjd)*24*3600     #interval in s

T_off = twindow - T_on                                  #interval in s

#EXPECTED BACKGROUND
RoI = 2.0     #deg

solid_angle_on = 2*np.pi*(1-np.cos(np.deg2rad(RoI)))          #srad
solid_angle_off = 2*np.pi*(np.sin(np.deg2rad(elevation_max))-np.sin(np.deg2rad(elevation_min)))      #srad

exp_background = N_off*(T_on/T_off)*(solid_angle_on/solid_angle_off)
exp_background_error = np.sqrt(N_off)*(T_on/T_off)*(solid_angle_on/solid_angle_off)

print("\nExpected background events: ", exp_background, " +/- ", exp_background_error, "events")

#ANGULAR DISTANCE BETWEEN EVENTS IN T_ON AND GRB (EQUATORIAL COORDINATES)
data_elevation_band_ON["AngularDistance_deg"] = np.rad2deg(
    np.arccos(
        np.sin(grb_radec.dec.rad)*np.sin(np.deg2rad(data_elevation_band_ON["Dec_J2000_deg"])) + 
        np.cos(grb_radec.dec.rad)*np.cos(np.deg2rad(data_elevation_band_ON["Dec_J2000_deg"]))*
        np.cos(np.deg2rad(data_elevation_band_ON["RA_J2000_deg"] - grb_radec.ra.deg))
        )
    )

print(data_elevation_band_ON)


#New dataframe with all events inside RoI
data_signal = data_elevation_band_ON[data_elevation_band_ON["AngularDistance_deg"]<2.0]

#Events in TEMPORAL and SPATIAL coincidence with GRB:
print(len(data_signal))





