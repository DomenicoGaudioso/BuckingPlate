# BuckingPlate

Web app professionale in **Python + Streamlit** per il **calcolo dell'instabilità elastica di piastre rettangolari in acciaio**.

## Funzioni principali
- piastra rettangolare con **w = 0** sui quattro bordi;
- vincoli rotazionali: **semplice / fisso / elastico** con **Kr** e **J**;
- piastra isotropa o ortotropa (**βx, ηx, βy, ηy**), con smearing di irrigidimenti uguali e regolari;
- irrigidimenti longitudinali e trasversali: **general, flat bar, symmetrical flat bar, T, angle, trapezoid**;
- stress analitici (**σx** ai quattro angoli, **σy** uniforme + patch loading, **τ** uniforme);
- stress **meshed** via CSV con colonne `x,y,sigma_x,sigma_y,tau`;
- risoluzione del problema agli autovalori con **serie di Fourier seno-seno**;
- risultati: **φcr, σx,cr, σy,cr, τcr**, modi, superfici 3D e coefficienti **Aij**;
- anteprima pre-analisi con **simboli dei vincoli** e **irrigidimenti visibili**.

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
