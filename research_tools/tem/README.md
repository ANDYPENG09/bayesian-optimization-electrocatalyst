# TNLA — TEM Nanoparticle & Lattice Analyzer v0.1

Offline local Python/Streamlit prototype. From this folder:

```bash
pip install -r requirements.txt
python -m unittest -v
streamlit run app.py --server.address 127.0.0.1
```

- TIFF pages, OME/ImageJ physical scale metadata or manual calibration, rectangular ROI.
- HRTEM: Hann-windowed FFT, subpixel peak candidates, reciprocal frequency, d-spacing and FCC/tetragonal geometric plane candidates.
- TEM: background correction, threshold, connected components, optional watershed, ID exclusion and annotated overlay.
- ECD, mean, SD, CV, number percentiles, D32/D43, nearest centroid distance, projected number density, geometric metal area and dispersion.
- Export CSV, JSON, PNG; no network service required.

Use synthetic demos first. The original local implementation passed three synthetic core tests. Real instrument TIFFs, vendor metadata and Streamlit UI still need validation. Pixel calibration must be verified; one d-spacing cannot uniquely identify a phase. Image contrast cannot chemically identify particles. The spherical FCC dispersion estimate is not ECSA or chemisorption; verify size distributions across representative fields. No proprietary images or data included. Automated integration with Materials Research Copilot is planned.
