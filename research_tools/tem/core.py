"""Local TEM image analysis; geometry and candidates, not phase confirmation."""
import xml.etree.ElementTree as ET
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, morphology, measure, feature, segmentation

def load_tiff(file, page=0):
    with tifffile.TiffFile(file) as t:
        p=t.pages[page]; arr=p.asarray(); meta={'pages':len(t.pages),'dtype':str(arr.dtype),'shape':list(arr.shape),'calibration_nm':None,'calibration_source':None}
        if t.ome_metadata and len(t.series)==1:
            try:
                root=ET.fromstring(t.ome_metadata); pixels=next(e for e in root.iter() if e.tag.split('}')[-1]=='Pixels')
                def nm(axis):
                    value=float(pixels.attrib['PhysicalSize'+axis]); unit=pixels.attrib.get('PhysicalSize'+axis+'Unit','µm')
                    return value*{'nm':1,'µm':1000,'um':1000,'Å':.1,'pm':.001,'mm':1e6}[unit]
                x,y=nm('X'),nm('Y')
                if min(x,y)>0 and np.isclose(x,y,rtol=.001):meta.update(calibration_nm=x,calibration_source='OME PhysicalSizeX/Y')
            except (ValueError,KeyError,StopIteration,ET.ParseError):pass
        if meta['calibration_nm'] is None and t.is_imagej:
            try:
                unit=t.imagej_metadata.get('unit',''); factor={'nm':1,'um':1000,'µm':1000,'Å':.1}[unit]
                x=p.tags['XResolution'].value;y=p.tags['YResolution'].value
                sx=factor*x[1]/x[0];sy=factor*y[1]/y[0]
                if min(sx,sy)>0 and np.isclose(sx,sy,rtol=.001):meta.update(calibration_nm=sx,calibration_source='ImageJ physical unit + resolution')
            except (KeyError,TypeError,ZeroDivisionError,AttributeError):pass
    if arr.ndim==3 and arr.shape[-1] in (3,4):arr=arr[...,:3]@np.array([.2126,.7152,.0722])
    if arr.ndim!=2:raise ValueError('Select a two-dimensional grayscale/RGB TIFF page.')
    if not np.isfinite(arr).all():raise ValueError('Nonfinite pixels in image.')
    return arr.astype(float),meta

def norm(a):
    lo,hi=np.percentile(a,[1,99]);return np.clip((a-lo)/max(hi-lo,1e-12),0,1)

def spectrum(a,scale,dmin=.12,dmax=.5,max_peaks=12):
    if scale<=0 or not 0<dmin<dmax or min(a.shape)<16:raise ValueError('Invalid calibration, range or ROI.')
    h,w=a.shape; win=np.outer(np.hanning(h),np.hanning(w)); a=(a-np.mean(a))*win
    f=np.abs(np.fft.fftshift(np.fft.fft2(a))); log=np.log1p(f)
    fy=np.fft.fftshift(np.fft.fftfreq(h,scale));fx=np.fft.fftshift(np.fft.fftfreq(w,scale)); rad=np.hypot(fy[:,None],fx[None,:])
    eligible=(rad>=1/dmax)&(rad<=1/dmin)&(rad<=.9/(2*scale)); eligible &= (fy[:,None]>0)|((fy[:,None]==0)&(fx[None,:]>0))
    pts=feature.peak_local_max(log,min_distance=3,threshold_abs=float(np.median(log)+3*np.median(np.abs(log-np.median(log)))),exclude_border=1)
    pts=[(r,c) for r,c in pts if eligible[r,c]];pts.sort(key=lambda p:f[p],reverse=True)
    out=[]
    def delta(l,m,r):
        denom=l-2*m+r
        return float(np.clip(.5*(l-r)/denom,-.5,.5)) if denom<0 else 0.
    for r,c in pts[:max_peaks]:
        dx=delta(log[r,c-1],log[r,c],log[r,c+1]);dy=delta(log[r-1,c],log[r,c],log[r+1,c]);gx=fx[c]+dx/(w*scale);gy=fy[r]+dy/(h*scale);g=np.hypot(gx,gy)
        out.append({'x_fft_px':float(c+dx),'y_fft_px':float(r+dy),'gx_nm_inv':float(gx),'gy_nm_inv':float(gy),'d_nm':float(1/g),'normal_angle_deg':float(np.degrees(np.arctan2(gy,gx))%180),'fft_amplitude':float(f[r,c]),'bin_scale_d_nm':float(max(1/(w*scale),1/(h*scale))/g**2)})
    return log,out

def plane_candidates(d,a=.3923,c=None,structure='fcc',tol=.035):
    if min(d,a,tol)<=0 or (c is not None and c<=0):raise ValueError('Invalid lattice parameters.')
    c=a if c is None else c; out=[]
    for h in range(5):
      for k in range(h+1):
       for l in range(5):
        if not h+k+l:continue
        if structure=='fcc' and not(h%2==k%2==l%2):continue
        if structure=='fcc' and l>k:continue
        ref=1/np.sqrt((h*h+k*k)/a**2+l*l/c**2);err=abs(d-ref)/ref
        if err<=tol:out.append({'hkl':f'({h}{k}{l})','d_ref_nm':float(ref),'error_pct':float(err*100)})
    return sorted(out,key=lambda x:x['error_pct'])

def particles(a,scale,polarity='dark',background_nm=10,threshold=None,min_nm=.6,max_nm=20,split=False,seed_nm=1.):
    if not 0<min_nm<max_nm or scale<=0 or background_nm<=0:raise ValueError('Invalid particle settings.')
    if min(a.shape)<8:raise ValueError('ROI too small.')
    v=norm(a);bg=ndi.gaussian_filter(v,max(1,background_nm/scale));score=(bg-v) if polarity=='dark' else (v-bg)
    if np.ptp(score)<1e-12:return np.zeros(a.shape,int),[],0.
    t=float(filters.threshold_otsu(score)) if threshold is None else float(threshold)
    mask=score>t; mask=morphology.remove_small_objects(mask,min_size=max(1,int(np.pi*(min_nm/2/scale)**2*.5)))
    mask=ndi.binary_fill_holes(mask)
    if split:
        dist=ndi.distance_transform_edt(mask); peaks=feature.peak_local_max(dist,min_distance=max(1,int(seed_nm/scale)),labels=mask)
        markers=np.zeros(mask.shape,int)
        for i,(y,x) in enumerate(peaks,1):markers[y,x]=i
        labs=segmentation.watershed(-dist,markers,mask=mask) if len(peaks) else measure.label(mask)
    else:labs=measure.label(mask)
    rows=[]; overlay=np.zeros(labs.shape,int)
    for r in measure.regionprops(labs):
        y0,x0,y1,x1=r.bbox;d=r.equivalent_diameter_area*scale
        if y0==0 or x0==0 or y1==a.shape[0] or x1==a.shape[1] or not min_nm<=d<=max_nm:continue
        idx=len(rows)+1; overlay[labs==r.label]=idx
        rows.append({'id':idx,'ecd_nm':float(d),'area_nm2':float(r.area*scale**2),'major_nm':float(r.axis_major_length*scale),'minor_nm':float(r.axis_minor_length*scale),'x_px':float(r.centroid[1]),'y_px':float(r.centroid[0]),'solidity':float(r.solidity)})
    return overlay,rows,t

def statistics(rows,scale,area_px,density=21.45,a=.3923,accessible=1.):
    if not rows:return {}
    d=np.array([r['ecd_nm'] for r in rows]); n=len(d);mean=float(d.mean());sd=float(d.std(ddof=1)) if n>1 else None
    d32=float(sum(d**3)/sum(d**2));d43=float(sum(d**4)/sum(d**3))
    atom_volume=a**3/4;atom_area=np.sqrt(3)*a**2/4
    dispersion=100*accessible*6*atom_volume/atom_area/d32
    result={'N':n,'mean_nm':mean,'sample_sd_nm':sd,'CV_pct':100*sd/mean if sd is not None else None,'D10_number_nm':float(np.percentile(d,10)),'D50_number_nm':float(np.percentile(d,50)),'D90_number_nm':float(np.percentile(d,90)),'D32_nm':d32,'D43_nm':d43,'projected_count_per_nm2':n/(area_px*scale**2),'geometric_surface_m2_g':6000*accessible/(density*d32),'geometric_metal_dispersion_pct':float(dispersion) if dispersion<=100 else None,'dispersion_model_warning':'Continuum model unreliable for very small clusters; value >100% withheld.' if dispersion>100 else 'Spherical fcc approximation with (111) atomic area; not measured accessible dispersion.'}
    if n>1:
        xy=np.array([[r['x_px'],r['y_px']] for r in rows])*scale; dist,_=cKDTree(xy).query(xy,k=2);result['observed_mean_nearest_neighbor_center_nm']=float(dist[:,1].mean())
    return result
