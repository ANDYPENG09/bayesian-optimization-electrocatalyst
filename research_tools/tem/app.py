import io,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from core import load_tiff,norm,spectrum,plane_candidates,particles,statistics
st.set_page_config(page_title='TEM Nanoparticle & Lattice Analyzer',layout='wide')
st.title('TEM Nanoparticle & Lattice Analyzer')
st.caption('TNLA · AI4M/S · Local image analysis · 晶格 / 粒径 / 分散性')
st.info('自动识别结果均为候选，需人工复核。上传文件在本地 Streamlit 进程处理；请勿将服务暴露到公网。')
f=st.file_uploader('TEM TIFF / TIF',type=['tif','tiff'])
if f:
    import tifffile
    with tifffile.TiffFile(io.BytesIO(f.getvalue())) as t:count=len(t.pages)
    page=st.sidebar.number_input('TIFF page (zero-based)',0,count-1,0) if count>1 else 0
    try:a,meta=load_tiff(io.BytesIO(f.getvalue()),page)
    except Exception as e:st.error(str(e));st.stop()
else:
    y,x=np.indices((256,256)); a=np.ones_like(x,dtype=float); rng=np.random.default_rng(5)
    for cy,cx,r in [(40,40,10),(110,80,15),(180,150,20),(55,180,13),(200,50,12)]:a[(y-cy)**2+(x-cx)**2<=r*r]=.25
    a+=rng.normal(0,.025,a.shape);meta={'calibration_nm':.1,'calibration_source':'Synthetic demo','shape':list(a.shape)}
    st.caption('未上传图像：显示合成颗粒示例。切换合成晶格示例可测试 FFT。')
    if st.checkbox('Synthetic HRTEM fringe demo'):a=np.cos(2*np.pi*x*.025/.226)+.05*rng.normal(size=x.shape);meta={'calibration_nm':.025,'calibration_source':'Synthetic lattice demo','shape':list(a.shape)}
with st.sidebar:
    st.header('Calibration / 标定')
    st.json(meta)
    method=st.radio('标定方式',['元数据 / 手动 nm/px','标尺长度与像素距离'])
    if method=='标尺长度与像素距离':
        length=st.number_input('Scale bar (nm)',min_value=.001,value=10.0,format='%.5f');p=st.number_input('Scale bar length (px)',min_value=1.,value=100.);scale=length/p
    else:scale=st.number_input('nm per pixel',min_value=.000001,value=float(meta.get('calibration_nm') or .1),format='%.6f')
    confirmed=st.checkbox('确认物理标定正确',value=not bool(f))
    st.caption('无可信元数据时，默认值仅为占位；必须用原始标尺校准。')
    st.header('ROI / 分析区域')
    h,w=a.shape
    if min(h,w)<16:st.error('图像过小');st.stop()
    x0,x1=st.slider('X (px)',0,w,(0,w));y0,y1=st.slider('Y (px)',0,h,(0,h))
    if min(x1-x0,y1-y0)<16:st.error('ROI 至少 16 px × 16 px');st.stop()
    st.write(f'{scale:.6g} nm/px')
roi=a[y0:y1,x0:x1]
st.image(norm(roi),caption=f'ROI {x1-x0} × {y1-y0} px',clamp=True,width=450)
if not confirmed:st.warning('确认标定后才进行定量分析。');st.stop()
cal={'nm_per_pixel':scale,'source':'user-confirmed '+method,'roi_xyxy':[x0,y0,x1,y1],'image':f.name if f else 'synthetic_demo','page':page if f else 0}
t1,t2=st.tabs(['HRTEM 晶格分析','TEM 颗粒统计'])
with t1:
    c1,c2,c3=st.columns(3)
    dmin=c1.number_input('d min (nm)',.001,10.,.12);dmax=c2.number_input('d max (nm)',.002,20.,.5);tol=c3.number_input('匹配容差 (%)',.1,30.,3.5)/100
    c1,c2,c3=st.columns(3);structure=c1.selectbox('Reference lattice',['fcc','tetragonal (geometric only)']);aa=c2.number_input('a (nm)',.01,5.,.3923,format='%.5f');cc=c3.number_input('c (nm)',.01,5.,.3700,format='%.5f',disabled=structure=='fcc')
    st.caption('FCC 默认 a=0.3923 nm (Pt)。四方晶系仅几何候选，不处理具体空间群或结构因子。')
    try:
        log,peaks=spectrum(roi,scale,dmin,dmax)
        fig,ax=plt.subplots(figsize=(5,5));ax.imshow(log,cmap='gray');ax.set_title('FFT · candidate spots')
        for i,p in enumerate(peaks,1):ax.scatter(p['x_fft_px'],p['y_fft_px'],facecolors='none',edgecolors='red',s=55);ax.text(p['x_fft_px']+3,p['y_fft_px'],str(i),color='yellow',fontsize=8)
        ax.set_axis_off();st.pyplot(fig);plt.close(fig)
        table=[]
        for i,p in enumerate(peaks,1):
            matches=plane_candidates(p['d_nm'],aa,None if structure=='fcc' else cc,'fcc' if structure=='fcc' else 'tetragonal',tol)
            table.append({'id':i,**p,'candidates':'; '.join(f"{m['hkl']} ({m['error_pct']:.2f}%)" for m in matches)})
        st.dataframe(pd.DataFrame(table),use_container_width=True)
        st.download_button('Export lattice CSV',pd.DataFrame(table).to_csv(index=False),'lattice.csv','text/csv')
        st.download_button('Export lattice JSON',json.dumps({'calibration':cal,'reference':{'a_nm':aa,'c_nm':aa if structure=='fcc' else cc,'structure':structure,'relative_tolerance':tol},'peaks':table},indent=2),'lattice.json','application/json')
        st.caption('FFT 自动峰可能来自噪声、载体、莫尔条纹或重叠晶体；需结合真实空间条纹、多个反射与相组成复核。bin_scale_d_nm 是频率网格对应的间距尺度，不是统计置信区间。')
    except ValueError as e:st.warning(str(e))
    st.subheader('Real-space cross-check / 实空间复核')
    length=st.number_input('多周期条纹总距离 (px)',min_value=1.,value=50.); periods=st.number_input('完整间隔数',min_value=1,value=5);st.write(f'd = {length*scale/periods:.5f} nm')
with t2:
    c1,c2,c3=st.columns(3);pol=c1.selectbox('Contrast',['dark','bright']);mi=c2.number_input('Min ECD (nm)',.01,100.,.6);ma=c3.number_input('Max ECD (nm)',.02,1000.,20.)
    bg=st.number_input('Background Gaussian sigma (nm)',.05,1000.,10.);auto=st.checkbox('Otsu automatic threshold',value=True);th=None if auto else st.slider('Contrast threshold',0.,1.,.2,.005)
    split=st.checkbox('Optional watershed (review every split)',value=False);seed=st.number_input('Watershed seed distance (nm)',.01,100.,1.)
    try:labs,rows,threshold=particles(roi,scale,pol,bg,th,mi,ma,split,seed)
    except ValueError as e:st.warning(str(e));st.stop()
    st.caption(f'{len(rows)} candidate objects · threshold {threshold:.4f} · border-touching objects excluded')
    rejected=st.multiselect('Exclude false detections by ID / 排除误识别',options=[r['id'] for r in rows]);rows=[r for r in rows if r['id'] not in rejected]
    fig,ax=plt.subplots(figsize=(7,7));ax.imshow(norm(roi),cmap='gray')
    from skimage.segmentation import find_boundaries
    mask=np.where(np.isin(labs,rejected),0,labs);edges=find_boundaries(mask,mode='inner')&(mask>0);ov=np.zeros((*labs.shape,4));ov[edges]=[1,.3,.1,1];ax.imshow(ov)
    for r in rows:ax.text(r['x_px'],r['y_px'],str(r['id']),color='yellow',fontsize=7)
    ax.set_axis_off();st.pyplot(fig);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=180,bbox_inches='tight');plt.close(fig)
    dens=st.number_input('Metal density (g/cm³; Pt default)',.01,30.,21.45);a_geo=st.number_input('FCC lattice a (nm; Pt default)',.01,2.,.3923,format='%.5f');access=st.slider('Assumed accessible surface fraction',0.,1.,1.)
    result=statistics(rows,scale,roi.size,dens,a_geo,access)
    st.json(result)
    if rows:
        fig,ax=plt.subplots(figsize=(6,3));ax.hist([r['ecd_nm'] for r in rows],bins='auto',edgecolor='black');ax.set(xlabel='Equivalent circular diameter (nm)',ylabel='Particle count');st.pyplot(fig);plt.close(fig)
    st.dataframe(pd.DataFrame(rows),use_container_width=True)
    st.download_button('Export particle CSV',pd.DataFrame(rows).to_csv(index=False),'particles.csv','text/csv')
    st.download_button('Export statistics JSON',json.dumps({'calibration':cal,'settings':{'polarity':pol,'background_sigma_nm':bg,'threshold':threshold,'min_nm':mi,'max_nm':ma,'watershed':split,'seed_nm':seed,'excluded_ids':rejected,'density_g_cm3':dens,'fcc_a_nm':a_geo,'accessible_fraction':access},'statistics':result,'particles':rows},indent=2),'tem_statistics.json','application/json')
    st.download_button('Export annotated PNG',buf.getvalue(),'particles_overlay.png','image/png')
    st.caption('颗粒识别基于投影反差，不能确认元素；重叠颗粒、碳载体纹理、非均匀背景需人工审核。几何分散度/面积不是实测 ECSA 或化学吸附分散度。')
