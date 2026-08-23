from pathlib import Path
import json
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
RAW=REPO/'data/raw'
OUT=REPO/'results/erie'
OUT.mkdir(parents=True,exist_ok=True)
MOD=RAW/'LE_CHL_MODIS_SQ_6e88_c718_3d53.csv'
FIELD=RAW/'ErieSummary_2008_2017.csv'

mod=pd.read_csv(MOD,parse_dates=['time (UTC)'])
times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique()))
lats=np.sort(mod['latitude (degrees_north)'].unique())
lons=np.sort(mod['longitude (degrees_east)'].unique())
nonmissing=int(pd.to_numeric(mod['chlorophyll (ug/L)'],errors='coerce').notna().sum())

raw=pd.read_csv(FIELD,encoding='latin1',header=1)
raw['date']=pd.to_datetime(raw['Date'],format='%m/%d/%y',errors='coerce')
raw['lat']=pd.to_numeric(raw['Lattitude'],errors='coerce')
raw['lon']=pd.to_numeric(raw['Longitude'],errors='coerce')
raw['chl']=pd.to_numeric(raw['Extracted Chlorophyll\n(µgChl-A/L)'],errors='coerce')
raw['cat']=raw['Sample Depth (category)'].astype(str).str.strip().str.lower()
f=raw[(raw.cat=='surface') & raw.date.dt.year.between(2014,2016) & raw.chl.notna() &
      raw.lat.between(lats.min(),lats.max()) & raw.lon.between(lons.min(),lons.max())].copy()

sat_days=times.tz_localize(None).normalize()
lags=[]; nearest=[]
for d in f.date:
    dd=np.abs((sat_days-d.normalize()).days)
    j=int(np.argmin(dd)); nearest.append(j); lags.append(int(dd[j]))
f['nearest_t']=nearest; f['abs_day_lag']=lags

station=f.groupby('Station').agg(n=('chl','size'),lat_median=('lat','median'),lon_median=('lon','median'),chl_median=('chl','median'),chl_max=('chl','max')).reset_index()
station.to_csv(OUT/'field_station_audit_v5.csv',index=False)

res={
 'modis_rows':int(len(mod)),
 'modis_snapshots':int(len(times)),
 'grid_nlat':int(len(lats)),
 'grid_nlon':int(len(lons)),
 'modis_nonmissing_chlorophyll':nonmissing,
 'field_surface_n':int(len(f)),
 'field_station_labels':int(f['Station'].nunique()),
 'field_sampling_dates':int(f['date'].dt.normalize().nunique()),
 'field_2014':int((f.date.dt.year==2014).sum()),
 'field_2015':int((f.date.dt.year==2015).sum()),
 'field_2016':int((f.date.dt.year==2016).sum()),
 'field_median_chlorophyll':float(f.chl.median()),
 'field_max_chlorophyll':float(f.chl.max()),
 'field_same_day':int((f.abs_day_lag==0).sum()),
 'field_within_1_day':int((f.abs_day_lag<=1).sum()),
 'field_within_2_days':int((f.abs_day_lag<=2).sum())
}
(OUT/'lake_erie_audit_v5.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
