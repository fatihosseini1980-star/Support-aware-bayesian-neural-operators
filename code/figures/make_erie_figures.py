import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.ndimage import label
from scipy.stats import wilcoxon
import os
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]; RAW=REPO/'data/raw'; OUT=REPO/'results/erie'; FIG=REPO/'figures'; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Tinos'],'font.size':9,'axes.titlesize':9.5,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
mod=pd.read_csv(str(RAW/'LE_CHL_MODIS_SQ_6e88_c718_3d53.csv'),parse_dates=['time (UTC)']); times=pd.DatetimeIndex(np.sort(mod['time (UTC)'].unique())); lats=np.sort(mod['latitude (degrees_north)'].unique()); lons=np.sort(mod['longitude (degrees_east)'].unique()); T,ny,nx=len(times),len(lats),len(lons)
A=np.full((T,ny,nx),np.nan,np.float32); it=pd.Categorical(mod['time (UTC)'],categories=times,ordered=True).codes; iy=np.searchsorted(lats,mod['latitude (degrees_north)']); ix=np.searchsorted(lons,mod['longitude (degrees_east)']); A[it,iy,ix]=mod['chlorophyll (ug/L)'].to_numpy(np.float32)
cnt=np.isfinite(A).sum(0); lab,n=label(cnt>=3); sizes=np.bincount(lab.ravel()); water=lab==(np.argmax(sizes[1:])+1); LON,LAT=np.meshgrid(lons,lats)
# geographic water mask
m0=Basemap(projection='cyl',llcrnrlon=float(lons.min())-.02,llcrnrlat=float(lats.min())-.02,urcrnrlon=float(lons.max())+.02,urcrnrlat=float(lats.max())+.02,resolution='i')
geo=np.array([[not m0.is_land(float(lo),float(la)) for lo in lons] for la in lats]); display=water&geo
# field data
raw=pd.read_csv(str(RAW/'ErieSummary_2008_2017.csv'),encoding='latin1',header=1); raw['date']=pd.to_datetime(raw['Date'],format='%m/%d/%y',errors='coerce'); raw['lat']=pd.to_numeric(raw['Lattitude'],errors='coerce'); raw['lon']=pd.to_numeric(raw['Longitude'],errors='coerce'); raw['chl']=pd.to_numeric(raw['Extracted Chlorophyll\n(µgChl-A/L)'],errors='coerce'); raw['cat']=raw['Sample Depth (category)'].astype(str).str.strip().str.lower(); f=raw[(raw.cat=='surface')&raw.date.dt.year.between(2014,2016)&raw.chl.notna()&raw.lat.between(lats.min(),lats.max())&raw.lon.between(lons.min(),lons.max())].copy(); st=f.groupby('Station')[['lat','lon']].median().reset_index()
# Overview time-mean log map
LA=np.log1p(A); valid=np.isfinite(LA); Z=np.divide(np.nansum(LA,axis=0),valid.sum(axis=0),out=np.full((ny,nx),np.nan),where=valid.sum(axis=0)>0); Z[~display]=np.nan
fig,ax=plt.subplots(figsize=(7.4,4.45))
m=Basemap(projection='cyl',llcrnrlon=float(lons.min())-.02,llcrnrlat=float(lats.min())-.02,urcrnrlon=float(lons.max())+.02,urcrnrlat=float(lats.max())+.02,resolution='i',ax=ax)
im=m.pcolormesh(LON,LAT,Z,latlon=True,shading='auto',cmap='viridis',rasterized=True); m.drawcoastlines(linewidth=.7,color='0.12'); m.drawparallels([41.4,41.6,41.8,42.0],labels=[1,0,0,0],linewidth=.15,color='0.82',dashes=[1,3]); m.drawmeridians([-83.4,-83.1,-82.8,-82.5],labels=[0,0,0,1],linewidth=.15,color='0.82',dashes=[1,3])
# hand-tuned label offsets in degrees
ofs={'WE2':(.012,.014),'WE4':(.012,.014),'WE6':(.012,-.016),'WE8':(.012,.014),'WE9':(-.055,.012),'WE12':(.012,.014),'WE13':(.012,.014),'WE14':(.012,.014),'WE15':(.012,.012),'WE2014.1':(-.012,-.035),'WE2014.2':(.012,-.035)}
for _,r in st.iterrows():
 one=str(r.Station).startswith('WE2014'); marker='^' if one else 'o'; ms=6.2 if one else 5.7
 ax.plot(r.lon,r.lat,marker=marker,ms=ms,mfc='white',mec='0.08',mew=1.0,zorder=6)
 dx,dy=ofs.get(str(r.Station),(.01,.01)); labeltxt=str(r.Station).replace('WE2014.','2014.')
 ax.text(r.lon+dx,r.lat+dy,labeltxt,fontsize=7.6,ha='left',va='center',zorder=7)
# legend proxies
ax.plot([],[],'o',ms=5.7,mfc='white',mec='0.08',mew=1,label='Recurring stations'); ax.plot([],[],'^',ms=6.2,mfc='white',mec='0.08',mew=1,label='One-off 2014 sites'); ax.legend(loc='lower left',frameon=False)
cb=fig.colorbar(im,ax=ax,fraction=.028,pad=.022); cb.set_label(r'Time-mean $\log(1+\mathrm{chlorophyll})$')
fig.tight_layout(); fig.savefig(str(FIG/'erie_data_overview_v5.pdf'),bbox_inches='tight'); fig.savefig(str(FIG/'erie_data_overview_v5.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
# Cross-fitted maps
pr=pd.read_csv(str(OUT/'crossfit/spatial_cv_predictions.csv')); # Representative dates are selected by a prespecified rule: among snapshots with
# at least 2,500 valid cells, choose the snapshot whose RMSE is closest to the
# annual median cross-fitted RMSE. The resulting time indices are 39, 70, 101.
idxs=[39,70,101]
obslist=[]; predlist=[]; errlist=[]; metrics=[]
for j in idxs:
 obs=np.log1p(A[j].astype(float)); predm=np.full((ny,nx),np.nan); g=pr[pr.t==j]; predm[g.iy.astype(int),g.ix.astype(int)]=g.pred.to_numpy(); full_target=g.target.to_numpy(); full_pred=g.pred.to_numpy(); full_rmse=float(np.sqrt(np.mean((full_pred-full_target)**2))); full_corr=float(np.corrcoef(full_pred,full_target)[0,1]); full_n=len(g); obs[~display]=np.nan; predm[~display]=np.nan; err=np.abs(predm-obs); obslist.append(obs); predlist.append(predm); errlist.append(err); metrics.append((full_rmse,full_corr,full_n))
vals=np.concatenate([z[np.isfinite(z)] for z in obslist+predlist]); vmin,vmax=np.percentile(vals,[1,99]); emax=np.percentile(np.concatenate([e[np.isfinite(e)] for e in errlist]),97)
fig,axs=plt.subplots(3,3,figsize=(8.4,7.6),sharex=True,sharey=True); letters=list('abcdefghi')
for r,j in enumerate(idxs):
 for c,D in enumerate([obslist[r],predlist[r],errlist[r]]):
  ax=axs[r,c]; mm=Basemap(projection='cyl',llcrnrlon=float(lons.min())-.015,llcrnrlat=float(lats.min())-.015,urcrnrlon=float(lons.max())+.015,urcrnrlat=float(lats.max())+.015,resolution='i',ax=ax)
  if c<2: im=mm.pcolormesh(LON,LAT,D,latlon=True,shading='auto',cmap='viridis',vmin=vmin,vmax=vmax,rasterized=True)
  else: ie=mm.pcolormesh(LON,LAT,D,latlon=True,shading='auto',cmap='magma',vmin=0,vmax=emax,rasterized=True)
  mm.drawcoastlines(linewidth=.55,color='0.10')
  if c==0: mm.drawparallels([41.4,41.6,41.8,42.0],labels=[1,0,0,0],linewidth=.01,color='0.85',fontsize=7)
  if r==2: mm.drawmeridians([-83.4,-83.1,-82.8,-82.5],labels=[0,0,0,1],linewidth=.01,color='0.85',fontsize=7)
  ax.text(.015,.965,f'({letters[3*r+c]})',transform=ax.transAxes,va='top',ha='left',fontsize=8.7,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=.8))
  if r==0: ax.set_title(['Observed MODIS','Cross-fitted prediction','Absolute error'][c])
  if c==0:
   rm,co,nn=metrics[r]; ax.text(-.22,.5,times[j].tz_localize(None).strftime('%Y-%m-%d'),transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=8.1)
  if c==1:
   rm,co,nn=metrics[r]; ax.text(.98,.03,f'RMSE={rm:.3f}\n' + rf'$r$={co:.3f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=7.1,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=1.2))
cax1=fig.add_axes([.18,.052,.46,.015]); cb1=fig.colorbar(im,cax=cax1,orientation='horizontal'); cb1.set_label(r'$\log(1+\mathrm{chlorophyll}\ [\mu\mathrm{g}/\mathrm{L}])$')
cax2=fig.add_axes([.72,.052,.18,.015]); cb2=fig.colorbar(ie,cax=cax2,orientation='horizontal'); cb2.set_label('Absolute error')
fig.subplots_adjust(left=.08,right=.985,top=.96,bottom=.105,wspace=.035,hspace=.055); fig.savefig(str(FIG/'erie_crossfitted_maps_v5.pdf'),bbox_inches='tight'); fig.savefig(str(FIG/'erie_crossfitted_maps_v5.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
# support-transfer summary + per-date CIs
summary=[]; perd=[]
base=str(OUT/'support_transfer_runs')
for bs in [9,13,17,21]:
 s=pd.read_csv(f'{base}/supportNN_bs{bs}_support_w128_seed13.csv'); c=pd.read_csv(f'{base}/supportNN_bs{bs}_centroid_w128_seed13.csv'); y=s.target.to_numpy()
 for meth,p in [('Support-aware',s.pred.to_numpy()),('PostAgg',c.post.to_numpy()),('Centroid',c.pred.to_numpy())]: summary.append(dict(block_width=bs,method=meth,rmse=float(np.sqrt(np.mean((p-y)**2))),mae=float(np.mean(np.abs(p-y))),corr=float(np.corrcoef(p,y)[0,1])))
 for t,idx in s.groupby('t').groups.items():
  ii=np.array(list(idx)); yy=s.loc[ii,'target'].to_numpy()
  for meth,pred in [('Support-aware',s.loc[ii,'pred'].to_numpy()),('PostAgg',c.loc[ii,'post'].to_numpy()),('Centroid',c.loc[ii,'pred'].to_numpy())]: perd.append(dict(block_width=bs,t=int(t),method=meth,rmse=float(np.sqrt(np.mean((pred-yy)**2)))))
sd=pd.DataFrame(summary); pd.DataFrame(perd).to_csv(str(OUT/'support_transfer_perdate_v5.csv'),index=False); sd.to_csv(str(OUT/'support_transfer_summary_v5.csv'),index=False)
# Pooled shifted-support RMSE with date-cluster bootstrap intervals.
# This uses the same estimand as the summary table; dates are resampled as clusters.
fig,ax=plt.subplots(figsize=(5.9,3.7)); markers={'Support-aware':'o','PostAgg':'s','Centroid':'^'}
rng=np.random.default_rng(20260823)
for meth in ['Support-aware','PostAgg','Centroid']:
 xs=[]; ms=[]; elo=[]; ehi=[]
 for bs in [9,13,17,21]:
  ss=pd.read_csv(f'{base}/supportNN_bs{bs}_support_w128_seed13.csv')
  cc=pd.read_csv(f'{base}/supportNN_bs{bs}_centroid_w128_seed13.csv')
  y=ss.target.to_numpy()
  pred={'Support-aware':ss.pred.to_numpy(),'PostAgg':cc.post.to_numpy(),'Centroid':cc.pred.to_numpy()}[meth]
  point=float(np.sqrt(np.mean((pred-y)**2)))
  dates=np.unique(ss.t.to_numpy()); boots=[]
  for _ in range(1000):
   samp=rng.choice(dates,size=len(dates),replace=True)
   idx=np.concatenate([np.flatnonzero(ss.t.to_numpy()==tt) for tt in samp])
   boots.append(float(np.sqrt(np.mean((pred[idx]-y[idx])**2))))
  lo,hi=np.quantile(boots,[.025,.975]); xs.append(bs); ms.append(point); elo.append(point-lo); ehi.append(hi-point)
 ax.errorbar(xs,ms,yerr=np.vstack([elo,ehi]),marker=markers[meth],lw=1.25,ms=4.8,capsize=2.5,label=meth)
ax.set_xlabel('Block width (native MODIS cells)'); ax.set_ylabel('Pooled shifted-support RMSE'); ax.set_xticks([9,13,17,21]); ax.grid(alpha=.18,linewidth=.5); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(str(FIG/'erie_support_transfer_v5.pdf'),bbox_inches='tight'); fig.savefig(str(FIG/'erie_support_transfer_v5.png'),dpi=300,bbox_inches='tight'); plt.close(fig)
# robustness tables
seedrows=[]
for seed in [13,29,47]:
 s=pd.read_csv(f'{base}/supportNN_bs13_support_w128_seed{seed}.csv'); c=pd.read_csv(f'{base}/supportNN_bs13_centroid_w128_seed{seed}.csv'); y=s.target.to_numpy(); seedrows.append(dict(seed=seed,support=float(np.sqrt(np.mean((s.pred-y)**2))),postagg=float(np.sqrt(np.mean((c.post-y)**2))),centroid=float(np.sqrt(np.mean((c.pred-y)**2)))))
pd.DataFrame(seedrows).to_csv(str(OUT/'support_transfer_seed_sensitivity_v5.csv'),index=False)
cap=[]
for w in [64,128,192]:
 s=pd.read_csv(f'{base}/supportNN_bs13_support_w{w}_seed13.csv'); c=pd.read_csv(f'{base}/supportNN_bs13_centroid_w{w}_seed13.csv'); y=s.target.to_numpy(); cap.append(dict(width=w,support=float(np.sqrt(np.mean((s.pred-y)**2))),postagg=float(np.sqrt(np.mean((c.post-y)**2))),centroid=float(np.sqrt(np.mean((c.pred-y)**2)))))
pd.DataFrame(cap).to_csv(str(OUT/'support_transfer_capacity_sensitivity_v5.csv'),index=False)
print('metrics dates',[(str(times[j]),*metrics[i]) for i,j in enumerate(idxs)])
print(sd.to_string(index=False))
