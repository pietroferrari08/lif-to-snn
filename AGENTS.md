# AGENTS.md — lif-to-snn

## Scopo del repository

Micro-progetto didattico di neuroscienze computazionali: partire da un neurone Leaky Integrate-and-Fire (LIF) implementato in Python, analizzare spike train, firing rate e ISI, costruire una piccola Spiking Neural Network (SNN) con PyTorch/snnTorch e addestrarla su un compito semplice.

**Priorità:** comprensione e autonomia dello studente > quantità di codice prodotto. Il progetto è extracurricolare: evitare ampliamenti non richiesti.

## Regola fondamentale: approvazione prima delle modifiche

- **Per impostazione predefinita lavora in sola lettura.** Puoi esaminare i file pertinenti, spiegare il codice e proporre interventi.
- Prima di creare, modificare, eliminare, rinominare o formattare file, presenta una proposta breve con: obiettivo, file interessati, cambiamenti previsti e metodo di verifica. **Attendi un'approvazione esplicita riferita a quella proposta.**
- Una richiesta di spiegazione, diagnosi o code review **non** autorizza modifiche. Non interpretare un generico «procediamo» come permesso illimitato a modificare il repository.
- Dopo l'approvazione, limita gli interventi ai file e allo scopo approvati. Se emergono modifiche ulteriori, interrompiti e chiedi una nuova autorizzazione.
- Non eseguire `git commit`, `git push`, cancellazioni, reset, installazioni di dipendenze, download o operazioni esterne senza un'autorizzazione separata e specifica.
- Dopo le modifiche, mostra il riepilogo del diff, i controlli eseguiti e gli eventuali limiti della verifica. Non dichiarare test riusciti senza averli effettivamente eseguiti.

## Modalità didattica

- L'utente sta imparando Python, programmazione a oggetti, tensori PyTorch, SNN e addestramento supervisionato.
- Quando l'utente vuole imparare o risolvere un esercizio, **non completare automaticamente l'esercizio**: chiarisci i concetti e offri indizi progressivi. Fornisci codice completo solo se richiesto o approvato.
- Spiega in modo operativo ogni nuova classe, funzione e metodo: che cosa crea, quali argomenti riceve, che cosa restituisce e quale shape hanno i tensori.
- Distingui i fatti verificati nel codice dalle ipotesi e dalle previsioni matematiche. Evidenzia gli errori e la loro causa (concettuale, programmazione, calcolo, interpretazione).
- Preferisci cambiamenti piccoli, nomi di variabili chiari, commenti pertinenti e implementazioni leggibili. Non introdurre architetture, astrazioni o librerie superflue.

## Convenzioni tecniche

- Usa l'ambiente virtuale locale `.venv` del repository. Su macOS il comando di esecuzione, se pertinente, è `./.venv/bin/python nome_file.py`.
- Non aggiungere `.venv/`, `__pycache__/`, `.DS_Store`, file temporanei, dati grandi o segreti a Git. Verifica le esclusioni in `.gitignore`.
- Per la simulazione LIF distinguere sempre potenziale prima del reset, stato dopo il reset e spike binario. Dichiarare se le frequenze sono in spike/time step oppure in Hz.
- Per confrontare il LIF manuale con `snn.Leaky`, controlla esplicitamente convenzioni di soglia, `reset_mechanism`, `reset_delay` e gestione dello stato. Non assumere equivalenza numerica senza verifica.
- Per le reti, distinguere chiaramente `nn.Linear` (trasformazione pesata) da `snn.Leaky` (dinamica spiking); verificare le shape di ingressi, spike, correnti e stati.
- Durante il training preservare il grafo di autograd: non usare `.item()`, `.detach()` o `torch.no_grad()` nel percorso di calcolo della loss e dei gradienti. Sono ammessi per stampe, diagnostica o inizializzazioni manuali, dove appropriato.
- Per esperimenti comparativi registrare configurazione, seed quando utile, metriche e limiti; distinguere training, validation e test.
- Per dettagli di API e comportamento usare prioritariamente la documentazione ufficiale PyTorch e snnTorch, verificando la versione installata quando necessario.

## Verifica e comunicazione

- Prima di un intervento controlla solo i file necessari e lo stato Git pertinente; evita ispezioni o refactoring generalizzati senza motivo.
- Se autorizzato a eseguire test, parti da controlli minimi e riproducibili; riferisci comando, risultato ed errori. Non modificare dipendenze per far passare un test senza una nuova approvazione.
- Quando una modifica è pronta, riporta: (1) cosa è cambiato, (2) perché, (3) come è stato verificato, (4) cosa lo studente dovrebbe saper spiegare, (5) eventuale prossimo passo.
- Il README documenta il progetto e i risultati; questo file documenta il metodo di collaborazione con Codex. Mantieni entrambi concisi.

## Direzione del progetto

Percorso minimo: LIF manuale → statistiche e spike train → rete feedforward snnTorch → training supervisionato di una piccola classificazione → valutazione e semplice esperimento su beta/soglia → README e figure riproducibili.

Estensioni come MNIST completo, tuning esteso, dataset biomedicali, STDP e hardware neuromorfico sono **fuori dallo scope minimo**: proporle separatamente e non iniziarle automaticamente.
