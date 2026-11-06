"""
API FastAPI per la predizione della struttura secondaria delle proteine (Q8).

Riceve una sequenza di aminoacidi e restituisce la struttura secondaria
predetta dal modello ResidualCNN addestrato (una lettera per residuo).

Le feature del profilo di sequenza (PSSM) sono impostate a zero, perche' l'API
riceve solo la sequenza di aminoacidi. Il modello e' stato addestrato con il
profilo, quindi le predizioni possono essere meno accurate dei risultati
riportati nel README.

Requisiti
---------
- Dipendenze installate:  pip install fastapi uvicorn
- Checkpoint presente in:  outputs/checkpoints/proposed_cnn.pt
  (si ottiene con:  python -m src.train --model proposed)

Avvio (dalla cartella principale del progetto, con il venv attivo)
------------------------------------------------------------------
    python -m uvicorn src.api:app

Poi aprire http://127.0.0.1:8000/docs: la documentazione interattiva permette
di provare gli endpoint dal browser (POST /predict -> "Try it out").

Endpoint
--------
    GET  /          messaggio di benvenuto e link alla documentazione
    GET  /health    stato del servizio, modello e dispositivo usato
    POST /predict   predizione Q8 per una sequenza

Esempio di richiesta (PowerShell)
---------------------------------
    Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
        -ContentType "application/json" `
        -Body '{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}'

Esempio di richiesta (Linux / macOS)
------------------------------------
    curl -X POST http://127.0.0.1:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}'

La risposta e' un JSON con la sequenza, la lunghezza e la stringa "structure"
(una lettera tra L, B, E, G, I, H, S, T per ogni residuo). Se la sequenza e'
vuota, contiene simboli non supportati o supera 700 residui, l'API risponde
con errore 422 e un messaggio esplicativo.
"""

from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from src.models import MODELS


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

# Lunghezza massima gestita: il modello e' stato addestrato su sequenze
# lunghe al massimo 700 residui.
MAX_LENGTH = 700

# Percorso del checkpoint del modello proposto (calcolato rispetto a questo file).
CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "checkpoints"
    / "proposed_cnn.pt"
)

# Aminoacidi accettati, nello stesso ordine dei 22 canali one-hot del dataset
# (il canale 21, NoSeq, e' il padding e non e' un aminoacido valido in input).
AMINO_ACIDS = [
    "A", "C", "E", "D", "G", "F", "I", "H", "K", "M", "L",
    "N", "Q", "P", "S", "R", "T", "W", "V", "Y", "X",
]

# Classi Q8 nell'ordine di uscita del modello (indice 0-7).
STRUCTURE_LABELS = ["L", "B", "E", "G", "I", "H", "S", "T"]

# Usa la GPU se disponibile, altrimenti la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Protein Secondary Structure Prediction API",
    description="Predict Q8 secondary structure from an amino acid sequence.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Validazione dell'input
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Corpo della richiesta di /predict: una sola stringa con la sequenza."""

    sequence: str

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        """Pulisce e valida la sequenza.

        Rimuove spazi e a capo, converte in maiuscolo e controlla che la
        sequenza non sia vuota, che contenga solo aminoacidi supportati e che
        non superi la lunghezza massima. In caso contrario FastAPI risponde
        con errore 422.
        """
        # Elimina qualsiasi spazio/a capo (utile se si incolla una sequenza
        # su piu' righe) e porta tutto in maiuscolo.
        sequence = "".join(value.split()).upper()

        if not sequence:
            raise ValueError("Sequence cannot be empty.")

        # Simboli presenti nella sequenza ma non nella lista degli aminoacidi.
        invalid = sorted(set(sequence) - set(AMINO_ACIDS))

        if invalid:
            raise ValueError(
                f"Unsupported amino acid symbols: {', '.join(invalid)}"
            )

        if len(sequence) > MAX_LENGTH:
            raise ValueError(
                f"Sequence too long: maximum length is {MAX_LENGTH} residues."
            )

        return sequence


# ---------------------------------------------------------------------------
# Caricamento del modello
# ---------------------------------------------------------------------------

def load_model():
    """Costruisce il modello proposto e carica i pesi dal checkpoint.

    Returns:
        Il modello in modalita' di valutazione (dropout disattivato).

    Raises:
        RuntimeError: Se il file del checkpoint non esiste.
    """
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = MODELS["proposed"]().to(device)

    # weights_only=True carica solo i pesi, senza eseguire codice arbitrario.
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()  # disattiva il dropout per la predizione

    return model


# Il modello viene caricato una sola volta, all'avvio del servizio, e riusato
# per tutte le richieste.
model = load_model()


# ---------------------------------------------------------------------------
# Codifica dell'input
# ---------------------------------------------------------------------------

def encode_sequence(sequence: str) -> torch.Tensor:
    """Costruisce il tensore di input (1, 46, 700) da una sequenza di aminoacidi.

    Le 46 feature per posizione sono, come nel training:
        canali 0-21   aminoacido (one-hot)
        canali 22-23  flag N-terminale e C-terminale
        canali 24-45  profilo di sequenza (qui sempre zero)

    Le posizioni oltre la fine della sequenza restano a zero (padding).

    Args:
        sequence: Sequenza validata, in maiuscolo.

    Returns:
        Tensore float32 di forma (1, 46, 700) sul dispositivo in uso.
    """
    x = torch.zeros((1, 46, MAX_LENGTH), dtype=torch.float32)

    # One-hot dell'aminoacido: per ogni posizione attiva il canale
    # corrispondente alla lettera.
    for position, amino_acid in enumerate(sequence):
        aa_index = AMINO_ACIDS.index(amino_acid)
        x[0, aa_index, position] = 1.0

    # Flag terminali: primo e ultimo residuo (verificato sui dati di training,
    # dove il flag N e' alla posizione 0 e il flag C all'ultimo residuo).
    x[0, 22, 0] = 1.0
    x[0, 23, len(sequence) - 1] = 1.0

    # I canali 24-45 (profilo di sequenza) restano a zero, perche' l'API
    # riceve solo la sequenza.

    return x.to(device)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Messaggio di benvenuto con il link alla documentazione interattiva."""
    return {
        "message": "Protein Secondary Structure Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Controllo di stato: se risponde, il modello e' stato caricato."""
    return {
        "status": "ok",
        "model": "proposed",
        "device": str(device),
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Predice la struttura secondaria Q8 di una sequenza di aminoacidi.

    Returns:
        JSON con la sequenza, la lunghezza e la stringa "structure", che ha
        una lettera (L, B, E, G, I, H, S, T) per ogni residuo.
    """
    sequence = request.sequence
    x = encode_sequence(sequence)

    # no_grad: niente calcolo dei gradienti, perche' stiamo solo predicendo.
    with torch.no_grad():
        logits = model(x)  # forma (1, 8, 700)

        # Classe con il punteggio piu' alto per ogni posizione; si tengono
        # solo le prime len(sequence) posizioni, scartando il padding.
        predicted_ids = logits.argmax(dim=1)[0, :len(sequence)].cpu().tolist()

    # Da indici numerici (0-7) a lettere della struttura secondaria.
    structure = "".join(
        STRUCTURE_LABELS[class_id] for class_id in predicted_ids
    )

    return {
        "sequence": sequence,
        "length": len(sequence),
        "structure": structure,
        "class_names": STRUCTURE_LABELS,
        "note": (
            "Sequence-profile features were set to zero because "
            "the API receives only the amino acid sequence. "
            "Predictions may differ from those obtained with full input features."
        ),
    }