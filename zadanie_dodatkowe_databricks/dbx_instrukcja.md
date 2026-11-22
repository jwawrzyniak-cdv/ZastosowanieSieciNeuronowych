# Zadanie dodatkowe: wdrożenie modelu w Databricks Free Edition

Charakter: nieobowiązkowe, wykonywane poza zajęciami. Wynik co najmniej 50 punktów na 100 podnosi średnią z laboratoriów o pół stopnia, ale nie powyżej 5,0
Czas: nie mierzyliśmy, zależy głównie od czasu budowy endpointu (dokumentacja Databricks podaje 10-25 minut na zmianę konfiguracji endpointu)
Środowisko: Google Colab (część 1) i Databricks Free Edition na serverless compute (część 2), Keras 3.13.2, w Databricks na backendzie PyTorch
Pliki: `dbx_przygotowanie_colab.ipynb`, `dbx_databricks.ipynb`

> Uwaga prowadzącego: notebook do Databricks nie był uruchamiany na Databricks. Przygotowałem go na podstawie dokumentacji z listopada 2026, a fragmenty niezależne od platformy (wczytanie modelu z TensorFlow na backendzie PyTorch, opakowanie pyfunc, zapis i odczyt modelu w MLflow) sprawdziłem lokalnie na tych samych wersjach bibliotek. Jeśli któryś krok nie zadziała, opiszcie dokładnie, co się stało. Rzetelny opis problemu też się liczy.

## Cel

Na laboratoriach model kończył życie w notebooku: trening, ewaluacja, wnioski. W praktyce to dopiero połowa drogi, bo model ma potem odpowiadać na zapytania innych systemów. W tym zadaniu przeprowadzicie sieć CNN dla CIFAR-10 przez typowy proces MLOps na platformie Databricks:

1. trening w Colabie i zapis modelu,
2. przeniesienie do innego środowiska i sprawdzenie, czy model liczy to samo,
3. zapis przebiegu w MLflow (parametry, metryka, artefakt),
4. rejestracja wersji modelu w Unity Catalog,
5. uruchomienie endpointu REST (Model Serving) i wysłanie do niego zapytania.

Przy okazji zobaczycie w praktyce, do czego służy wieloplatformowość Keras 3: model wytrenowany na TensorFlow zostanie wczytany i obsłużony przez PyTorch bez zmiany ani jednej linii jego definicji.

## Czym jest Databricks Free Edition i jakie ma ograniczenia

Free Edition to bezpłatna, ograniczona wersja platformy Databricks dla studentów i osób uczących się. Zastąpiła wycofaną w 2025 roku Community Edition. Rejestracja przez `https://login.databricks.com/?intent=CE_SIGN_UP`, logowanie kodem wysyłanym e-mailem, kontem Google albo Microsoft. Nie podaje się karty płatniczej.

Ograniczenia, które mają znaczenie w tym zadaniu (według strony „Databricks Free Edition limitations”, stan z 18.10.2026):

- Dostępne jest tylko obliczanie serverless, bez własnych klastrów i bez GPU w notebookach.
- Zużycie zasobów jest ograniczone dziennym limitem. Po jego przekroczeniu obliczenia są wyłączane do końca dnia, ale dane i ustawienia zostają.
- Endpointy Model Serving działają wyłącznie na CPU, a liczba aktywnych endpointów jest ograniczona (dokumentacja nie podaje liczby).
- Dostęp do internetu z notebooków jest ograniczony do listy zaufanych domen. Dlatego nie pobieramy danych w Databricks, tylko wgrywamy gotowe pliki do Volume.
- Konto służy wyłącznie do celów niekomercyjnych, a nieaktywne konta mogą zostać usunięte.

Środowisko serverless w wersji 6 ma Pythona 3.12, PyTorch 2.12 (CPU), mlflow-skinny 3.12, numpy 2.3 i pandas 2.3. Nie ma TensorFlow ani Keras, dlatego Keras doinstalowujemy i używamy backendu PyTorch.

## Przygotowanie

1. Załóżcie konto Databricks Free Edition. Wystarczy prywatny e-mail.
2. Zróbcie część 1 w Colabie albo przygotujcie model z laboratorium 2 (wymagania poniżej).

## Krok 1: model w Colabie

Otwórzcie `dbx_przygotowanie_colab.ipynb` w Google Colab, włączcie GPU i uruchomcie wszystkie komórki. Notebook trenuje małą sieć CNN (sześć warstw splotowych, około 300 tys. parametrów) i pobiera na Wasz komputer dwa pliki:

- `cifar_cnn.keras`, około 4 MB,
- `cifar_probka.npz`, około 0,6 MB: 200 obrazów testowych, ich etykiety i prawdopodobieństwa policzone w TensorFlow.

Jeśli chcecie wdrożyć własny model z laboratorium 2, ustawcie w sekcji 3 `UZYJ_WLASNEGO = True`. Model musi przyjmować obraz 32x32x3 w skali 0-255 (normalizacja w warstwie `Rescaling` wewnątrz modelu) i składać się wyłącznie ze standardowych warstw Keras. Warstwy augmentacji (`RandomFlip`, `RandomRotation` itp.) w trybie predykcji nic nie robią, więc mogą zostać.

## Krok 2: Volume w Unity Catalog

W Databricks dane i pliki trzymane są w Unity Catalog. Volume to katalog na pliki dowolnego typu, adresowany ścieżką `/Volumes/<katalog>/<schemat>/<wolumen>/`. W Free Edition macie gotowy katalog `workspace` ze schematem `default`.

1. Po lewej wybierzcie Catalog, rozwińcie `workspace` i kliknijcie schemat `default`.
2. Utwórzcie Volume o nazwie `zsn` (przycisk Create, Volume, typ Managed). Zamiast tego możecie w edytorze SQL wykonać `CREATE VOLUME IF NOT EXISTS workspace.default.zsn`.
3. Otwórzcie Volume i wgrajcie oba pliki (Upload to this volume).

## Krok 3: notebook w Databricks

1. Workspace, Import, wskażcie plik `dbx_databricks.ipynb`.
2. Otwórzcie notebook i podłączcie go do Serverless (lista Connect u góry).
3. W panelu Environment po prawej sprawdźcie wersję środowiska. Notebook przygotowaliśmy pod wersję 6.
4. Uruchamiajcie sekcje po kolei. Sekcja 0 restartuje Pythona po instalacji Keras, to normalne.

Sekcje 1-5 nie tworzą endpointu i powinny zająć kilka minut. Po sekcji 5 macie zarejestrowany model w Unity Catalog. Zobaczcie go w Catalog Explorer (`workspace.default.cifar_cnn`) razem z wersją, aliasem `champion` i sygnaturą. Przebieg MLflow znajdziecie w zakładce Experiments albo w ikonie kolby obok notebooka.

Sekcja 6 tworzy endpoint i czeka, aż będzie gotowy, sprawdzając jego stan co minutę. Jeśli budowa się nie powiedzie, otwórzcie Serving, kliknijcie endpoint i przejrzyjcie logi budowy kontenera (build logs). Najczęstsze przyczyny to brakujący pakiet albo konflikt wersji w `pip_requirements`.

Sekcja 8 usuwa endpoint. Nie pomijajcie jej: limit endpointów i dzienny limit zasobów obowiązują całe konto.

## Co oddajecie

1. Notebook `dbx_databricks.ipynb` wyeksportowany z Databricks z wynikami komórek (File, Export, IPython Notebook albo HTML).
2. Trzy zrzuty ekranu: przebieg w MLflow z parametrami i metryką, model w Catalog Explorer z wersją i aliasem oraz endpoint w stanie Ready (albo log błędu budowy, jeśli endpoint nie powstał).
3. W sekcji 9 notebooka odpowiedzi na pytania poniżej, 5-8 zdań łącznie albo krótko na każde.

Pytania:

1. Jaka była maksymalna różnica prawdopodobieństw między TensorFlow (Colab) a PyTorch (Databricks)? Czy zmieniła się któraś przewidywana klasa? Skąd biorą się takie różnice?
2. Dlaczego normalizacja pikseli jest warstwą modelu, a nie krokiem wykonywanym przed wywołaniem `predict`? Co by się stało, gdyby klient endpointu wysłał piksele w skali 0-1?
3. Do czego służy sygnatura modelu i dlaczego rejestr w Unity Catalog jej wymaga?
4. Ile trwało pierwsze, a ile kolejne zapytanie do endpointu? Jak wiąże się to z opcją `scale_to_zero_enabled`?
5. Wasz model ma obsługiwać 500 zapytań na sekundę z czasem odpowiedzi poniżej 100 ms. Co zmienilibyście w konfiguracji endpointu albo w samym modelu?

## Źródła

- Databricks, Free Edition limitations: `https://docs.databricks.com/aws/en/getting-started/free-edition-limitations`
- Databricks, Serverless environment version 6: `https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/six`
- Databricks, Create custom model serving endpoints: `https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints`
- Databricks, Deploy custom models (pyfunc): `https://docs.databricks.com/aws/en/machine-learning/model-serving/custom-models`
- MLflow, dokumentacja modułu `mlflow.pyfunc`: `https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html`
- Databricks, Debugging guide for Model Serving: `https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-debug`
- Keras, Multi-backend Keras: `https://keras.io/getting_started/`
