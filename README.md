# GGUF Knowledge Base Builder

Webowa aplikacja w Pythonie oparta o FastAPI i pojedynczy frontend HTML/JS. Pozwala wrzucac dokumenty przez przegladarke, budowac osobista baze wiedzy jako plik GGUF, wersjonowac dokumenty, wykrywac duplikaty po hashach oraz wykonywac semantyczne wyszukiwanie po chunkach.

<img width="1344" height="1029" alt="image" src="https://github.com/user-attachments/assets/4a799b25-dec4-4046-afb9-2e747f4f59b1" />
<img width="1507" height="489" alt="image" src="https://github.com/user-attachments/assets/8bf9b071-5636-45a2-bfbd-8ca2ebde7c7e" />


## Funkcje

- upload wielu plikow przez przegladarke
- upload archiwow `.zip` z automatycznym rozpakowaniem obslugiwanych plikow
- automatyczna kolejka buildow z historia jobow
- czesciowy rebuild tylko dla zmienionych dokumentow
- deduplikacja po `sha256` i wersjonowanie dokumentow
- zapis stanu w `data/kb_state.json`
- embeddingi `sentence-transformers/all-MiniLM-L6-v2`
- chunking bliski tokenizerowi modelu: 512 tokenow z overlapem 64
- filtrowanie wyszukiwania po dokumencie, zrodle, typie i tagach
- podglad chunkow i debug search w UI
- logowanie haslem, rate limiting i limit rozmiaru uploadu
- panel administracyjny: zmiana hasla, reset limiterow, czyszczenie historii buildow, diagnostyka
- OCR fallback dla PDF, jesli PyMuPDF ma dostep do OCR w runtime
- gotowy kontener Docker z offline runtime

## Zmienne srodowiskowe

- `KB_PASSWORD` - haslo do logowania w aplikacji. Domyslnie `change-me`.
- `KB_SESSION_HOURS` - czas zycia sesji cookie. Domyslnie `12`.
- `KB_MAX_UPLOAD_MB` - maksymalny rozmiar pojedynczego pliku. Domyslnie `64`.
- `KB_RATE_LIMIT_PER_MINUTE` - liczba requestow na minute z jednego hosta. Domyslnie `120`.
- `KB_ENABLE_PDF_OCR` - `1` lub `0`, wlacza OCR fallback dla PDF.
- `KB_PDF_OCR_LANGS` - jezyki OCR dla PyMuPDF, domyslnie `eng`.
- `KB_TEST_MODE` - lekki tryb testowy bez ladowania prawdziwego modelu embeddingow.

## Uruchomienie lokalne

1. Zainstaluj zaleznosci:

```bash
pip install -r requirements.txt
```

2. Ustaw haslo aplikacji:

```bash
export KB_PASSWORD='silne-haslo'
```

3. Uruchom aplikacje:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Otworz przegladarke:

```bash
http://localhost:8000
```

## Testy

```bash
KB_TEST_MODE=1 python -m unittest discover -s tests -v
```

## Endpointy

- `GET /health`
- `GET /favicon.ico`
- `GET /auth/session`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /admin/diagnostics`
- `POST /admin/password`
- `POST /admin/clear-builds`
- `POST /admin/reset-rate-limits`
- `POST /upload`
- `POST /build`
- `GET /status`
- `GET /builds`
- `GET /documents`
- `GET /document/{id}/chunks`
- `DELETE /document/{id}`
- `GET /download`
- `GET /search`

## Budowanie obrazu

```bash
docker build -t knowledge-base .
```

## Uruchomienie przez Docker Compose

```bash
docker-compose up -d
```

Po starcie zaloguj sie haslem ustawionym w `KB_PASSWORD`. Jesli nie zmienisz nic w compose, domyslne haslo to `change-me`.

## Przeniesienie na maszyne offline

1. Zapisz obraz:

```bash
docker save knowledge-base | gzip > knowledge-base.tar.gz
```

2. Skopiuj plik `knowledge-base.tar.gz` na docelowa maszyne.

3. Zaladuj obraz:

```bash
docker load < knowledge-base.tar.gz
```

4. Uruchom kontener:

```bash
docker-compose up -d
```

## Dostep z innego hosta

1. Znajdz IP hosta:

```bash
ip a
```

lub

```bash
hostname -I
```

2. Otworz aplikacje z innej maszyny:

```bash
http://<IP_HOSTA>:8000
```

3. Jesli firewall blokuje port:

```bash
ufw allow 8000
```

## Struktura projektu

```text
.
|-- app/
|   |-- services/
|   |   |-- kb.py
|   |   |-- kb_base.py
|   |   |-- kb_builds.py
|   |   `-- kb_documents.py
|   |-- templates/
|   |   `-- index.html
|   |-- config.py
|   |-- main.py
|   |-- models.py
|   `-- routes.py
|-- data/
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- download_model.py
|-- main.py
`-- requirements.txt
```

## Uwagi

- W runtime kontener dziala offline. Model embeddingow jest pobierany do obrazu na etapie build.
- Dane dokumentow, stan bazy i pliki GGUF sa trzymane w `./data`, dzieki czemu przetrwaja restart kontenera.
- OCR dla PDF zalezy od mozliwosci OCR udostepnionych przez lokalne srodowisko PyMuPDF.

## Licencja

MIT. Szczegoly sa w pliku `LICENSE`.
