Ecco una proposta per un file `README.md` professionale, tecnico e accademico, generato sulla base del codice e dei file forniti.

-----

# EBPlateLite v2.1

Un'applicazione web avanzata per l'analisi dell'instabilità elastica (buckling) di piastre in acciaio irrigidite e non irrigidite. Il software adotta un approccio ibrido, fornendo un solutore semianalitico ad alta precisione, un solutore agli elementi finiti (FEM) equivalente e un modulo per le verifiche manuali secondo l'Eurocodice 3.

## 📖 Indice

  - [Caratteristiche Principali](https://www.google.com/search?q=%23caratteristiche-principali)
  - [Fondamenti Teorici](https://www.google.com/search?q=%23fondamenti-teorici)
  - [Architettura degli Input e Output](https://www.google.com/search?q=%23architettura-degli-input-e-output)
  - [Requisiti di Sistema](https://www.google.com/search?q=%23requisiti-di-sistema)
  - [Installazione](https://www.google.com/search?q=%23installazione)
  - [Esempio di Utilizzo](https://www.google.com/search?q=%23esempio-di-utilizzo)
  - [Licenza](https://www.google.com/search?q=%23licenza)

## ✨ Caratteristiche Principali

  - **Solutore Semianalitico (Tipo EBPlate):** Calcolo dei moltiplicatori critici di instabilità $\phi_{cr}$ tramite il metodo di Ritz con serie di Fourier.
  - **Solutore FEM Equivalente:** Modellazione agli elementi finiti tramite OpenSeesPy (elementi ShellMITC4) utilizzando un approccio a sottodomini equivalenti ("smearing") per gli irrigidimenti.
  - **Verifiche Manuali EC3:** Calcolo automatico dei parametri di instabilità secondo EN 1993-1-5 (es. $k_\sigma$, slellezza $\bar{\lambda}_p$, fattore di riduzione $\rho$ e larghezza efficace $b_{eff}$).
  - **Supporto Irrigidimenti Complessi:** Gestione di sezioni aperte (piatti, angolari, a T) e chiuse (trapezi, cassoni), con mesh FEM conforme alle bande irrigidenti.
  - **Visualizzazione Avanzata:** Anteprima geometrica 2D interattiva e rendering 3D dei modi di instabilità (autovettori) tramite Plotly.

## 🔬 Fondamenti Teorici

Il nucleo computazionale dell'applicazione si basa sulla teoria classica dell'instabilità elastica delle piastre sottili (teoria di Kirchhoff-Love).

### Metodo Semianalitico (Serie di Ritz-Fourier)

Il problema dell'instabilità viene formulato ricercando il minimo dell'energia potenziale totale del sistema $\Pi = U - W$, dove $U$ è l'energia di deformazione elastica flessionale e $W$ è il lavoro compiuto dalle tensioni membranali nel piano.
L'abbassamento trasversale $w(x,y)$ viene approssimato tramite una doppia serie di funzioni di forma (es. seni e coseni) che rispettano le condizioni al contorno geometriche. Il problema variazionale viene tradotto nel seguente problema agli autovalori generalizzato:
$$[R_0] \{c\} = \lambda [R_G] \{c\}$$
Dove $[R_0]$ è la matrice di rigidezza flessionale (che include i termini di ortotropia $D_x, D_y$ derivanti dalla presenza di irrigidimenti) e $[R_G]$ è la matrice geometrica dipendente dal campo di tensioni nel piano ($\sigma_x, \sigma_y, \tau$). L'autovalore minimo positivo $\lambda_{min}$ (definito nell'app come $\phi_{cr}$) rappresenta il moltiplicatore critico dei carichi.

### Metodo FEM Equivalente

Il backend secondario utilizza una discretizzazione spaziale basata su elementi ShellMITC4 per mitigare i fenomeni di *shear locking*. Gli irrigidimenti sono modellati spalmando le loro proprietà ("smearing") su una larghezza di banda efficace ("band width"), definendo uno spessore equivalente membranale ($t_{eq,mem}$) e flessionale ($t_{eq,bend}$) per garantire l'equivalenza energetica.

### Verifiche Normative (EN 1993-1-5)

Il calcolo valuta la suscettibilità all'instabilità locale calcolando la tensione critica euleriana $\sigma_{cr} = k_\sigma \frac{\pi^2 E}{12(1-\nu^2)} \left(\frac{t}{b}\right)^2$. A partire da questa, si valuta la snellezza adimensionale $\bar{\lambda}_p = \sqrt{f_y / \sigma_{cr}}$ e si ricava il coefficiente di riduzione $\rho$, che permette di definire la larghezza efficace $b_{eff}$ del pannello compresso o inflesso.

## ⚙️ Architettura degli Input e Output

L'interfaccia utente richiede l'inserimento granulare di parametri chimico-fisici e geometrici, elaborando risultati sia tabellari che grafici.

### Input Attesi

  * **Parametri Piastra:** Larghezza ($a$), Altezza ($b$), Spessore ($t$), Modulo di Young ($E$), Coefficiente di Poisson ($\nu$), Tensione di snervamento ($f_y$).
  * **Condizioni al Contorno:** Vincoli rotazionali (Semplice/Incastro/Elastico) e rigidezze torsionali ai bordi ($K_r$) per tutti e quattro i lati.
  * **Tensioni nel Piano:** Tensioni normali lineari ai bordi ($\sigma_x, \sigma_y$), carichi concentrati tipo *patch loading* e tensioni tangenziali ($\tau$). Supporto per l'importazione di campi di tensione da mesh esterne via CSV.
  * **Irrigidimenti (Stiffeners):** Array di oggetti definiti per orientamento (longitudinale/trasversale), tipologia (es. closed box, trapezoid, flat bar), coordinate di posizionamento e dimensioni della sezione trasversale.

### Output Generati

  * **Anteprima (Pre-Analisi):** Rendering vettoriale del dominio 2D e della mesh FEM conforme agli irrigidimenti.
  * **Risultati Semianalitici:** Moltiplicatore critico ($\phi_{cr}$), tensioni critiche ($\sigma_{x,cr}, \sigma_{y,cr}, \tau_{cr}$), dataset degli autovettori e grafico 3D del modo di instabilità dominante.
  * **Risultati FEM:** Moltiplicatore di carico $\lambda_{cr}$ estratto tramite l'algoritmo di Lanczos in OpenSees, mappa degli spostamenti nodali (Gradi di Libertà) e log di convergenza.
  * **Tabelle EC3:** Dettaglio del coefficiente di buckling $k_\sigma$, della soglia di snellezza $\bar{\lambda}_{lim}$ e delle frazioni di larghezze efficaci ($b_{e1}, b_{e2}$) per ogni sub-pannello interno calcolato.

## 💻 Requisiti di Sistema

Il software dipende dalle seguenti librerie Python:

  * `streamlit>=1.31.0`
  * `pandas>=1.5.0`
  * `numpy>=1.24.0`
  * `plotly>=5.18.0`
  * `scipy>=1.10.0`
  * `openseespy>=3.5.0`

## 🚀 Installazione

1.  Clonare il repository locale.
2.  Si raccomanda la creazione di un ambiente virtuale (es. `venv` o `conda`).
3.  Eseguire l'installazione delle dipendenze:
    ```bash
    pip install -r requirements.txt
    ```
4.  Avviare l'applicazione web:
    ```bash
    streamlit run app.py
    ```

## 🛠️ Esempio di Utilizzo

Una volta avviata l'interfaccia via Streamlit, l'utente può definire un nuovo caso manualmente tramite la barra laterale oppure importare un caso salvato in formato JSON.

1.  **Configurazione:** Impostare una piastra 3000x1500 mm, spessore 10 mm.
2.  **Carichi:** Applicare una $\sigma_x$ di compressione uniforme di 100 MPa sui bordi sinistro e destro.
3.  **Analisi:** Cliccare su *Calcola instabilità elastica (solver semianalitico)*.
4.  **Verifica:** Spostarsi nel tab "Risultati EBPlate" per consultare il valore di $\phi_{cr}$ e visualizzare il modo di sbandamento 3D. Controllare il tab "Verifiche manuali EC3" per il confronto con l'Eurocodice 3.
5.  **Esportazione:** Utilizzare il pulsante *Scarica caso JSON* a fine pagina per salvare la sessione.

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT License.
Copyright (c) 2026 Domenico Gaudioso.
Il software è fornito "COSÌ COM'È", senza alcuna garanzia esplicita o implicita. In nessun caso gli autori saranno responsabili per danni o altre responsabilità derivanti dall'utilizzo del software.

-----

**Domanda di revisione:** Questo livello di approfondimento teorico soddisfa le tue aspettative, o preferisci che aggiunga dettagli specifici su come vengono calcolate matematicamente le rigidezze torsionali degli irrigidimenti chiusi e aperti?
