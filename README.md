# ego-cit-mv

Minimal, reproducible toy experiments for MV-CIT (Critique-and-Iterative-Transformation) showing identity/welfare preservation under task pressure.

## Contents
- `notebooks/mv_cit_toy.ipynb`: Colab notebook used to generate the V5b results.
- `src/mv_cit_toy.py`: Standalone script reproducing the multi-task benchmark.
- `results/`: exported CSV/TXT summaries.

## Quick start (local)

```bash
pip install torch numpy
python src/mv_cit_toy.py
Outputs are written under results/.

Reproduce (local, isolated venv)
python -m venv .venv

# Windows:
.venv\Scripts\activate

pip install torch numpy
python src/mv_cit_toy.py
Run in Colab (optional)
Open notebooks/mv_cit_toy.ipynb in Google Colab (upload the notebook or open it from GitHub).

****

# ego-cit-mv

Esperimenti minimali e riproducibili per MV-CIT (Critique-and-Iterative-Transformation) che mostrano la preservazione dell’identità/benessere sotto pressione di compito.

## Contenuti:
notebooks/mv_cit_toy.ipynb: notebook Colab utilizzato per generare i risultati della versione V5b.

- src/mv_cit_toy.py: script standalone che riproduce il benchmark multi-task.
- results/: riepiloghi esportati in formato CSV/TXT.

## Quick start (local)

```bash
pip install torch numpy
python src/mv_cit_toy.py
Gli output vengono salvati nella cartella results/.

Riproduzione (locale, ambiente virtuale isolato)
python -m venv .venv

# Windows:
.venv\Scripts\activate

pip install torch numpy
python src/mv_cit_toy.py
Esecuzione in Colab (opzionale)
Apri notebooks/mv_cit_toy.ipynb in Google Colab (carica il notebook oppure aprilo direttamente da GitHub).
