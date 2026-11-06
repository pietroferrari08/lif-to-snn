# LIF-to-SNN — Computational Neuroscience

Un percorso pratico di neuroscienze computazionali: dall'implementazione manuale di un neurone **Leaky Integrate-and-Fire (LIF)** alla costruzione e al primo addestramento di una piccola **Spiking Neural Network (SNN)** con PyTorch e snnTorch.

> **Stato al 6 novembre 2026:** simulatore LIF, analisi degli spike, rete feedforward 2→1, introduzione al training e primi esperimenti completati. **La classificazione supervisionata di pattern/MNIST non è ancora implementata.** Il progetto ha finalità didattiche, non costituisce un modello biologico validato né un risultato di ricerca originale.

## 1. Obiettivo e domanda guida

**Domanda:** come un neurone LIF converte ingressi in spike, e come una rete di questi neuroni può apprendere a distinguere pattern?

Il percorso sviluppa progressivamente queste competenze:

1. Ricostruire l'equazione LIF e implementarne la discretizzazione.
2. Misurare spike train, firing rate e inter-spike interval (ISI).
3. Utilizzare tensori, classi e moduli PyTorch/snnTorch per collegare neuroni.
4. Comprendere loss, gradienti, optimizer, *surrogate gradients* e backpropagation through time (BPTT).
5. Addestrare e valutare una SNN su input diversi (fase successiva), documentando risultati e limiti.

L'obiettivo del microprogetto è mantenere codice **leggibile, riproducibile e spiegabile**. Per gli interventi assistiti da Codex, vedere [`AGENTS.md`](AGENTS.md).

## 2. Struttura del repository

```text
lif-to-snn/
├── 01_lif_neuron.py        # LIF manuale, spike train, ISI e sweep degli ingressi
├── 03_simple_snn.py        # Due LIF iniziali → connessione lineare → LIF di output
├── 04_training_basics.py   # Esempio minimo di training con nn.Linear, MSE e SGD
├── 05_snn_training.py      # Primo training della SNN su un firing rate target
├── README.md               # Questa documentazione
├── AGENTS.md               # Regole di collaborazione con Codex
├── .gitignore
└── .venv/                  # Ambiente virtuale locale, escluso da Git
```

I nomi rispecchiano i file sviluppati finora; **non esiste un `02_spike_trains.py` separato**, perché l'analisi degli spike è stata integrata in `01_lif_neuron.py`. La cartella `.venv` va ricreata localmente e non committata.

## 3. Ambiente ed esecuzione

**Requisiti:** Python 3, [PyTorch](https://pytorch.org/), [snnTorch](https://snntorch.readthedocs.io/), Matplotlib. Il simulatore iniziale può richiedere anche NumPy a seconda della versione del codice locale. Il progetto è stato sviluppato su macOS tramite VS Code e un ambiente `.venv`.

Da terminale, nella radice del repository:

```bash
# Creazione dell'ambiente, se non esiste già
python3 -m venv .venv

# Installazione delle dipendenze utilizzate nelle varie fasi
./.venv/bin/python -m pip install numpy matplotlib torch snntorch

# Esecuzione dei singoli esperimenti
./.venv/bin/python 01_lif_neuron.py
./.venv/bin/python 03_simple_snn.py
./.venv/bin/python 04_training_basics.py
./.venv/bin/python 05_snn_training.py
```

Per controllare le versioni effettivamente installate:

```bash
./.venv/bin/python -m pip show torch snntorch matplotlib numpy
```

> Le versioni dei pacchetti **non sono ancora fissate** in un `requirements.txt`: prima di pubblicare una versione riproducibile del progetto, occorre verificarle nell'ambiente funzionante e registrarle. Non sono necessari download di dataset per i quattro script attuali.

## 4. Modello matematico LIF implementato a mano

### 4.1 Dinamica

Una forma dell'equazione continua di membrana, ispirata a un circuito RC, è

\[
\tau_m \frac{dU}{dt}=-U+RI.
\]

Con Eulero esplicito e ingresso opportunamente riscalato si ottiene il modello discreto adottato nel simulatore:

\[
U_{k+1}=\beta U_k+J_k,
\quad \beta=1-\frac{\Delta t}{\tau_m},
\quad J_k=\frac{\Delta t}{\tau_m}RI_k.
\]

- `U`: potenziale di membrana (stato del neurone).
- `BETA` / `beta`: quota del potenziale precedente conservata a ogni passo.
- `J`: contributo dell'ingresso al potenziale.
- `THRESHOLD`: soglia di generazione dello spike.
- `STEP`: numero di passi temporali simulati.

Il simulatore manuale rileva uno spike quando `U >= THRESHOLD` e applica **reset a zero** (`U = 0`). Registra separatamente il potenziale **prima** e **dopo** il reset. Non è stata attribuita una durata fisica al singolo time step: i risultati sono quindi espressi per passo, non in Hz.

### 4.2 Esempio numerico

Per `BETA = 0.8`, `J = 0.4`, soglia 1 e stato iniziale nullo, i primi potenziali prima del reset sono:

```text
Step 1: 0.4000
Step 2: 0.7200
Step 3: 0.9760
Step 4: 1.1808 → spike e reset a zero
```

Gli spike successivi compaiono ai passi 8, 12, ... in condizioni di ingresso costante.

### 4.3 Spike train, firing rate e ISI

Uno *spike train* è la sequenza binaria degli eventi di spiking. Nel programma vengono utilizzate, fra le altre, le variabili `spikes`, `spike_times` e `spike_potentials`.

\[
N_{\mathrm{spike}}=\sum_k s_k,
\qquad r=\frac{N_{\mathrm{spike}}}{\texttt{STEP}}.
\]

Per passare agli Hz è necessario conoscere la durata `dt` di ogni passo:

\[
r_{\mathrm{Hz}}=\frac{N_{\mathrm{spike}}}{\texttt{STEP}\cdot dt}.
\]

Gli **ISI** sono le differenze fra gli istanti di spike consecutivi. L'ISI medio non è definito se gli spike osservati sono meno di due. Su una finestra temporale finita il firing rate non coincide necessariamente con l'inverso dell'ISI medio.

### 4.4 Sweep dell'ingresso

Simulazioni manuali con `BETA = 0.8`, soglia 1 e `STEP = 100`:

| Ingresso `J` | Spike | Rate (spike/step) | ISI medio (step) |
|---:|---:|---:|---:|
| 0.15 | 0 | 0.00 | Non definito |
| 0.25 | 12 | 0.12 | 8 |
| 0.40 | 25 | 0.25 | 4 |
| 0.60 | 50 | 0.50 | 2 |

Per ingresso costante, senza reset e con `0 < beta < 1`, il potenziale tende a

\[
U_{\infty}=\frac{J}{1-\beta}.
\]

Per `J = 0.15` risulta `U_inf = 0.75`, inferiore alla soglia: non si osservano spike partendo da zero.

Le visualizzazioni sviluppate comprendono potenziali pre/post-reset, spike train e confronto degli ingressi. La presenza e la formattazione delle singole linee grafiche dipendono dallo stato attuale dello script (per esempio la linea orizzontale della soglia è stata rimossa da un grafico in una modifica non ancora committata).

## 5. Prima rete SNN: `03_simple_snn.py`

Architettura feedforward:

```text
x = [1, 1]
     │
     ▼
Correnti pesate con [0.25, 0.60]  (pesi fissi di ingresso)
     │
     ▼
2 neuroni LIF, beta=0.8 e soglia=1
     │  spike s1(t), s2(t)
     ▼
    nn.Linear(2, 1, bias=False)  (due pesi sinaptici)
     │
     ▼
1 neurone LIF di output
     │
     ▼
Spike train di uscita e firing rate
```

Per ogni passo temporale si eseguono queste operazioni:

```python
current = x * weights
spk_in, mem = lif(current, mem)
current_out = connection(spk_in)
spk_out, mem_out = lif_out(current_out, mem_out)
```

- `snn.Leaky(...)` istanzia un modulo neurale LIF.
- `nn.Linear(2, 1, bias=False)` istanzia un modulo che applica due pesi sinaptici, senza bias.
- `connection(spk_in)` calcola una corrente pesata, **non** uno spike.
- `lif_out(current_out, mem_out)` aggiorna il neurone finale e restituisce spike e stato.
- `mem` (due elementi) e `mem_out` (un elemento) conservano stati di membrana distinti.

La configurazione sperimentata per la rete utilizza `beta=0.8`, `threshold=1`, `reset_mechanism="zero"` e `reset_delay=False`. Il simulatore manuale e snnTorch non devono essere considerati equivalenti in ogni dettaglio di implementazione: confrontare sempre le convenzioni su soglia e reset.

### 5.1 Risultati osservati

Con 30 time step, pesi di ingresso `[0.25, 0.60]` e connessione iniziale `[[0.6, 0.6]]`:

| Neurone | Spike | Rate (spike/step) |
|---|---:|---:|
| LIF 1 | 3 | 0.100 |
| LIF 2 | 15 | 0.500 |
| LIF output | 7 | 0.233 |

Il neurone di uscita può generare spike anche quando soltanto uno dei due neuroni precedenti emette uno spike, grazie all'integrazione temporale del potenziale residuo.

### 5.2 Sweep del secondo peso sinaptico

Tenendo fissi **i pesi di ingresso** `[0.25, 0.60]` e il **primo peso sinaptico** a `0.6`, sono state confrontate due condizioni:

| Secondo peso sinaptico | Spike LIF 1 | Spike LIF 2 | Spike output | Rate output |
|---:|---:|---:|---:|---:|
| 0.99 | 3 | 15 | 7 | 0.2333 |
| 1.01 | 3 | 15 | 15 | 0.5000 |

Con peso `0.99`, lo spike singolo del secondo neurone non basta a superare la soglia da zero: il neurone di uscita integra eventi successivi. Con `1.01`, lo stesso evento può provocare immediatamente uno spike.

**Lezione metodologica:** `weights` (ingresso → primi LIF) e `connection.weight` (primi LIF → output) sono gruppi di pesi diversi. Cambiare il primo altera gli spike presinaptici; cambiare il secondo non può alterare gli spike del livello precedente in una rete feedforward.

## 6. Training: `04_training_basics.py` e `05_snn_training.py`

### 6.1 Training di una connessione lineare (senza spike)

Nel primo esercizio è stata addestrata una connessione con input `[1, 0]`, pesi iniziali `[0.2, 0.2]`, target `0.8`, loss MSE e ottimizzatore SGD.

```python
optimizer.zero_grad()       # pulisce i gradienti precedenti
prediction = connection(x)  # forward
loss = loss_fn(prediction, target)
loss.backward()             # accumula gradienti nei parametri
optimizer.step()            # modifica i pesi
```

Il parametro `connection.weight.grad` è il collegamento fra `backward()` e `optimizer.step()`: i due operano sugli **stessi** oggetti `nn.Parameter`.

Risultati osservati:

| Learning rate | Risultato |
|---|---|
| `0.1` | Primo peso da `0.2` a circa `0.6034` dopo 5 step; secondo peso invariato a `0.2` |
| `0.5` | Primo peso raggiunge `0.8` in un singolo aggiornamento, poi loss nulla |

Questa convergenza immediata a `lr=0.5` è una proprietà del caso quadratico semplice, non una garanzia generale di SGD.

### 6.2 Training preliminare della rete spiking

Nel secondo esercizio il compito non era classificare immagini, ma avvicinare il firing rate del neurone di uscita al target `0.30` (9 spike su 30 time step).

Per ogni epoca:

1. Reinizializzare potenziali `mem`, `mem_out` e lista degli spike di uscita.
2. Simulare 30 time step.
3. Conservare gli spike come **tensori**, senza `.item()` nel percorso della loss.
4. Calcolare `prediction = torch.stack(spikes_out_tensors).mean()`.
5. Calcolare la loss MSE rispetto a `target_rate`.
6. Eseguire `optimizer.zero_grad()`, `loss.backward()` e `optimizer.step()` nell'ordine appropriato (azzeramento prima del backward).

Nel modulo `snn.Leaky`, i **surrogate gradients** consentono di propagare gradienti attraverso la funzione a soglia: il forward produce spike discreti, mentre il backward utilizza una derivata sostitutiva. La propagazione attraverso gli stati successivi implementa la logica di **BPTT**.

Per i pesi di ingresso `[0.25, 0.60]`, i pesi della connessione iniziali `[[0.6, 0.6]]` e SGD con `lr=0.5`, la rete ha mostrato, in varie epoche, due comportamenti:

| Spike output | Rate | MSE verso 0.30 |
|---:|---:|---:|
| 7 | 0.2333 | 0.00444 |
| 15 | 0.5000 | 0.04000 |

I pesi si aggiornano, ma il firing rate può restare costante per più epoche e poi saltare bruscamente. **Il training non è convergente in questo esperimento**: l'osservazione illustra una difficoltà dell'ottimizzazione di risposte spiking discrete, non una prova di impossibilità per ogni combinazione di pesi.

### 6.3 Limiti dell'esperimento di training

- Un solo ingresso ripetuto e un target di firing rate fisso: non è ancora classificazione.
- Finestra finita di 30 passi; il rate varia in incrementi di `1/30`.
- L'ottimizzazione mediante surrogate gradients non garantisce miglioramento monotono della loss osservata.
- I primi pesi di ingresso sono tensori fissi: l'ottimizzatore aggiorna i pesi registrati in `connection.parameters()`.
- Senza più esempi e un set di valutazione separato non si può parlare di generalizzazione o accuratezza di classificazione.

## 7. Risultati, figure e riproducibilità

I risultati numerici riportati sono quelli discussi e verificati durante le sessioni del progetto; per riprodurli occorre rispettare le condizioni iniziali, i parametri dei neuroni e le impostazioni degli script. **Non è ancora presente una suite automatizzata di test** e non sono state fissate le versioni delle dipendenze.

Figure già sviluppate nel percorso: potenziale pre/post-reset, spike train e confronto dei valori di ingresso. Una raccolta finale delle figure in `figures/` e il loro inserimento nel README sono ancora da fare.

Per qualsiasi nuovo esperimento, registrare almeno: domanda, file eseguito, pesi iniziali, `beta`, soglia, durata, learning rate (se presente), risultati e conclusione. Modificare **una variabile alla volta** quando si vuole attribuire un effetto causale a un parametro.

## 8. Stato del progetto e prossimi passi

| Fase | Stato |
|---|---|
| Setup locale e Git | Repository inizializzato, due commit LIF/spike train confermati |
| LIF manuale | Completato |
| Spike train, firing rate, ISI | Completato |
| Prima SNN 2→1 | Completato |
| Comprensione e primi esperimenti di training | Completati a livello introduttivo |
| Classificazione supervisionata su input distinti | **Da fare** |
| MNIST binario 0 vs 1 | **Da fare** |
| Esperimenti sistematici su `beta` e soglia | **Da completare** |
| Figure finali e pubblicazione | **Da fare** |

**Prossima milestone:** costruire un piccolo problema di classificazione con input sintetici distinti; addestrare, misurare la prestazione su esempi non usati nel training e poi passare, se opportuno, a MNIST 0 vs 1.

Estensioni facoltative (non obbligatorie): tutte le dieci cifre MNIST, confronto di più architetture, studi sistematici di codifica e parametri. Il progetto deve rimanere proporzionato agli altri impegni accademici.

## 9. Riferimenti e documentazione

- [Documentazione Python](https://docs.python.org/3/)
- [PyTorch — `nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch — Optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [snnTorch — `snn.Leaky`](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html)
- [snnTorch — Training SNN](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)

Diario didattico e decisioni progettuali: [Notion — LIF to SNN](https://app.notion.com/p/4041fbe3b4af81dc934fd25711d12752?pvs=204).
