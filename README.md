
# EBPlateLite v1.6

Pacchetto completo **attualmente implementato** con:

- backend **semianalitico** tipo EBPlate;
- backend **FEM equivalente** con **scikit-fem**;
- preview della geometria e del modello FEM **prima del run**;
- tab di **confronto metodi**;
- supporto a irrigidimenti **aperti** e **chiusi** (`closed_section` / `closed box`);
- per il backend FEM: 
  - mesh conforme alle bande irrigidenti;
  - sottodomini plate equivalenti con `t_eq_mem` e `t_eq_bend`;
  - controllo di **connettività nodale** sui due bordi per irrigidimenti chiusi.

## Nota tecnica
Questo è il pacchetto completo **implementato oggi** nel perimetro delle librerie già usate nel progetto. Il backend FEM resta un modello **equivalente a sottodomini plate** e non un shell solver completo Gmsh + FEniCSx-Shells.

## Avvio
```bash
pip install -r requirements.txt
streamlit run app.py
```
