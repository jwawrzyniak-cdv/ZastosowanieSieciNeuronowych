# Laboratorium 6: Mechanizm uwagi i transformery w przetwarzaniu tekstu

Dane: PolEmo 2.0-IN z benchmarku KLEJ (5783 recenzje treningowe, 723 walidacyjne, 722 testowe, 4 klasy wydźwięku), te same co w laboratorium 5; w zadaniach rozszerzonych dla chętnych dodatkowo test PolEmo 2.0-OUT (494 recenzje z innych dziedzin)
Środowisko: Google Colab z GPU (T4), Keras 3 na backendzie torch, transformers (model `allegro/herbert-base-cased`), scikit-learn

## Cel

W laboratorium 5 sieci uczyły się języka wyłącznie z niecałych 6 tys. recenzji. Embeddingi startowały z losowych wartości, a słowa spoza słownika zamieniały się w jeden token OOV. Dziś sięgniecie po model, który przed zadaniem przeczytał bardzo duży korpus polskich tekstów: HerBERT, polski odpowiednik BERT. Zadanie pozostaje to samo, czyli wydźwięk recenzji w czterech klasach (minus, plus, amb, zero), więc wynik będzie można bezpośrednio porównać z TF-IDF, Embedding + GAP i BiLSTM.

Zanim użyjecie gotowego modelu, policzycie ręcznie w numpy to, co jest w nim najważniejsze: mechanizm uwagi (scaled dot-product attention). Potem zobaczycie, jak tokenizer subword HerBERT dzieli polskie słowa na fragmenty i dlaczego prawie nie ma tokenów spoza słownika. Główną część zajęć zajmuje fine-tuning całego HerBERT do klasyfikacji recenzji i porównanie go z modelami z laboratorium 5: nie tylko pod względem jakości, ale też liczby parametrów, czasu treningu i czasu predykcji.

Przygotujcie wyniki z laboratorium 5 na teście PolEmo 2.0-IN (accuracy, macro F1, liczba parametrów, czas treningu) dla TF-IDF + LR i dla sieci, które trenowaliście. Wpiszecie je do tabeli w zadaniu 4.

Fine-tuning w zadaniu 3 to najdłuższe obliczenia na tych zajęciach (czasu w Colabie nie mierzyliśmy, zależy też od przydzielonej karty). Uruchomcie go jak najwcześniej i w trakcie treningu wróćcie do odpowiedzi na pytania z zadań 1 i 2. Zadania A i B na końcu instrukcji to zadania rozszerzone.

## Przed zajęciami

Przeczytajcie `lab06_wprowadzenie.md`. Wyjaśnia mechanizm uwagi, budowę transformera, model HerBERT, tokenizację subword i fine-tuning.

Otwórzcie `lab06_szablon.ipynb` w Google Colab i włączcie GPU: Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU. Bez GPU fine-tuning HerBERT w czasie zajęć jest niewykonalny: przy próbach na zwykłym komputerze bez GPU jeden krok treningu (16 recenzji) trwał od kilku do kilkudziesięciu sekund, a epoka na pełnym zbiorze ma 362 kroki.

Ten notebook działa na Keras 3 z backendem torch, a nie TensorFlow jak laboratoria 1 do 5. Biblioteka transformers w wersji 5 nie obsługuje już TensorFlow, a HerBERT nie ma gotowego presetu w KerasHub. Backend wybiera się zmienną `os.environ["KERAS_BACKEND"] = "torch"` przed pierwszym `import keras` i nie da się go zmienić w trakcie sesji. Zacznijcie więc od świeżego środowiska wykonawczego. Jeśli w tej samej sesji Colaba uruchamialiście wcześniej inny notebook z Keras (np. z laboratorium 5), zrestartujcie sesję (Środowisko wykonawcze, Uruchom ponownie sesję; w angielskim interfejsie Runtime, Restart session). Komórka importów sprawdza `keras.backend.backend()` i zatrzymuje notebook czytelnym komunikatem, jeśli backend nie jest torch.

Pakiety (keras, torch, transformers, scikit-learn) są w Colabie preinstalowane. Przy pierwszym uruchomieniu Colab pobierze z Hugging Face tokenizer i wagi HerBERT (plik około 650 MB), co może potrwać chwilę. Komunikat o braku `HF_TOKEN` możecie zignorować, model jest publiczny.

Wszystkie parametry, od których zależy czas obliczeń (`MAX_LEN`, `BATCH`, `EPOKI`, `LR`, rozmiary podzbiorów), są w jednej komórce konfiguracyjnej. Zostawcie `TRYB_SKROCONY = False`. Tryb skrócony służy tylko do sprawdzenia, czy notebook przechodzi od początku do końca na komputerze bez GPU, i daje wyraźnie gorsze wyniki. Wczytanie danych jest takie samo jak w laboratorium 5 (te same pliki CSV, to samo mapowanie etykiet na minus, plus, amb, zero). Gotowe są funkcje `ocen`, `tabela_wynikow`, `przewidz`, `rysuj_historie`, `rysuj_macierz` i nowa `rysuj_uwage`.

## Zadanie 1: Uwaga w numpy

Scaled dot-product attention to softmax(QK^T / sqrt(d_k)) V, gdzie Q = X W_Q, K = X W_K, V = X W_V, a X to macierz wektorów tokenów (jeden wiersz na token).

1. Gotowa komórka tworzy sześć „tokenów” („hotel”, „był”, „czysty”, „ale”, „drogi” i token paddingu) z losowymi wektorami o wymiarze 8, macierze W_Q, W_K, W_V o wymiarach 8 × 4, macierz wyjściową W_O o wymiarach 4 × 8 i maskę, w której ostatni token to padding.
2. Napiszcie `softmax(z)` (po ostatniej osi, z odjęciem maksimum dla stabilności numerycznej) i `uwaga(Q, K, V, maska)`, która zwraca wyjście i macierz wag. Pozycjom paddingu w kluczach wpiszcie przed softmaxem bardzo małą liczbę (np. -1e9).
3. Sprawdźcie kształty Q, K, V, wag i wyjścia, to, że każdy wiersz wag sumuje się do 1, i to, że kolumna paddingu ma wagi 0. Narysujcie macierz wag funkcją `rysuj_uwage`.
4. Utwórzcie `keras.layers.MultiHeadAttention(num_heads=1, key_dim=4, use_bias=False)`, wywołajcie ją raz, żeby utworzyła wagi, i wpiszcie do niej te same macierze przez `set_weights` (kształty wag sprawdźcie w `get_weights()`). Porównajcie jej wyjście z `uwaga(...) @ W_O` i jej wagi uwagi (`return_attention_scores=True`) z Waszymi.
5. Dla d_k równego 4, 64 i 512 wylosujcie z rozkładu normalnego 1000 zapytań q i 10 kluczy k. Policzcie oceny q·k dla wszystkich par i porównajcie ich odchylenie standardowe oraz średnią największą wagę softmaxu (po 10 kluczach) bez dzielenia przez sqrt(d_k) i z dzieleniem.

Po co dzielimy przez sqrt(d_k)? Co się dzieje z gradientem softmaxu, gdy jedna waga jest bliska 1, a reszta bliska 0? Czym różni się macierz wag uwagi od wag warstwy Dense: co jest tu parametrem uczonym, a co zależy od wejścia? Dlaczego bez maski token paddingu mógłby wpływać na wynik?

## Zadanie 2: Tokenizacja subword HerBERT

1. Wczytajcie tokenizer `AutoTokenizer.from_pretrained("allegro/herbert-base-cased")` (gotowa komórka). Zobaczcie rozmiar słownika i tokeny specjalne `<s>`, `</s>`, `<pad>`, `<unk>`, `<mask>` z ich indeksami.
2. Stokenizujcie (`tokenizer.tokenize`) zdania z listy `ZDANIA`: odmianę słowa „lekarz”, zdanie bez polskich znaków, zdanie z literówkami i bardzo rzadkie słowo. Dla jednego zdania pokażcie pełne kodowanie z tokenami specjalnymi, paddingiem do długości 16 i `attention_mask`.
3. Policzcie odsetek tokenów `<unk>` na walidacji i średnią liczbę tokenów na słowo (słowa rozdzielone spacjami). Porównajcie z odsetkiem OOV na walidacji ze słownika z laboratorium 5.
4. Narysujcie histogram długości recenzji treningowych w tokenach HerBERT (bez obcinania), podajcie percentyle 50, 90, 95, 99 i odsetek recenzji dłuższych niż 128, 256 i 512 tokenów. Ostrzeżenie tokenizera o sekwencji dłuższej niż 512 tokenów jest tu nieszkodliwe, bo modelowi podamy sekwencje obcięte do `MAX_LEN`.
5. Napiszcie `koduj(teksty)`, która zwraca listę `[input_ids, attention_mask]`: dwie tablice int32 o wymiarach (liczba recenzji, `MAX_LEN`), z paddingiem i obcięciem (`padding="max_length"`, `truncation=True`, `max_length=MAX_LEN`). Zakodujcie `train_ft`, `valid_ft` i `test_df` do `X_train`, `X_valid`, `X_test`.

Dlaczego tokenizer subword prawie nie ma tokenów spoza słownika? Co oznacza `</w>` na końcu tokenu i dlaczego „lekarz” i „lekarzowi” dzielą się inaczej? Czy `MAX_LEN = 256` to rozsądny wybór, skoro HerBERT przyjmuje do 512 tokenów, i co tracimy przy obcięciu? Ile kosztuje dwukrotne wydłużenie sekwencji w samej warstwie uwagi?

## Zadanie 3: Fine-tuning HerBERT

Gotowa komórka definiuje moduł PyTorch `EnkoderHerBERT`. Wczytuje on `allegro/herbert-base-cased` przez `AutoModel` (z `add_pooling_layer=False`) i zwraca wektor pierwszego tokenu `<s>` (odpowiednik `[CLS]` w BERT) z ostatniej warstwy: 768 liczb na recenzję. Przy wczytaniu transformers wypisuje raport o wagach `cls.*` oznaczonych jako UNEXPECTED. To głowa używana w pretreningu do przewidywania zamaskowanych słów; nie korzystamy z niej, więc tak ma być.

`keras.layers.TorchModuleWrapper` zamienia moduł PyTorch w warstwę Keras: parametry modułu stają się wagami modelu, a `model.fit` trenuje je razem z resztą sieci. Warstwa przekazuje modułowi jeden argument, dlatego oba wejścia podajemy jako listę `[input_ids, attention_mask]`, a `forward` ją rozpakowuje.

1. Napiszcie `zbuduj_herbert()`: dwa wejścia `keras.Input(shape=(MAX_LEN,), dtype="int32")` o nazwach `input_ids` i `attention_mask`, warstwa `TorchModuleWrapper(EnkoderHerBERT(), output_shape=(None, 768))` wywołana na liście obu wejść, `Dropout(0.1)`, `Dense(4, activation="softmax")`. Kompilacja: `keras.optimizers.AdamW(learning_rate=LR)`, strata `sparse_categorical_crossentropy`, metryka accuracy.
2. Wyświetlcie `summary()`. Ile parametrów ma cały model, a ile sama głowa klasyfikacyjna?
3. Wytrenujcie model: `fit` na `X_train`, `y_train_ft` z walidacją na `X_valid`, `y_valid_ft`, `epochs=EPOKI`, `batch_size=BATCH`, `verbose=2` i callback `EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)`. Zmierzcie czas treningu i narysujcie historię (`rysuj_historie`). Jeśli Colab zgłosi brak pamięci GPU, zmniejszcie `BATCH` do 8 i zbudujcie model od nowa.
4. Oceńcie model na walidacji i, jeden raz, na teście (`ocen("HerBERT", ...)`, predykcje przez `przewidz`). Dla testu wypiszcie `classification_report` (z nazwami klas) i narysujcie macierz pomyłek (`rysuj_macierz`).

Modelu nie zapisujcie na dysk Colaba: `save_weights` zapisuje również stan optymalizatora AdamW, a plik ma wtedy około 1,5 GB.

Dlaczego learning rate jest tu około 50 razy mniejszy niż domyślny w Adam (0,001)? Co mogłoby się stać z wiedzą z pretreningu przy dużym learning rate? Dlaczego wystarczą 2 do 3 epoki? Jak wypadła klasa amb w porównaniu z laboratorium 5?

## Zadanie 4: Porównanie z laboratorium 5 i analiza błędów

1. Wpiszcie do słownika `WYNIKI_LAB5` własne wyniki z laboratorium 5 na teście PolEmo 2.0-IN: accuracy, macro F1, liczbę parametrów (dla TF-IDF + LR liczbę wag regresji) i czas treningu. Zestawcie je w jednej tabeli z HerBERT.
2. Zmierzcie czas predykcji jednej recenzji przez HerBERT: batch złożony z jednej recenzji, jedno wywołanie na rozgrzewkę, potem kilka powtórzeń i mediana.
3. Wypiszcie kilka recenzji z klasy amb, które HerBERT zaklasyfikował błędnie, z prawdopodobieństwami wszystkich klas (pierwsze 400 znaków). Policzcie, z którymi klasami amb jest mylona najczęściej.

Czy kilkadziesiąt razy więcej parametrów przełożyło się na proporcjonalnie lepszy wynik? Kiedy w praktyce wybralibyście TF-IDF z regresją logistyczną, a kiedy HerBERT (pomyślcie o sprzęcie, czasie odpowiedzi i liczbie recenzji do przetworzenia)? Czy błędy na amb to błędy modelu, czy raczej niejednoznaczne etykiety?

## Pytania kontrolne

1. Co oznaczają Q, K i V w mechanizmie uwagi i dlaczego wagi uwagi, w przeciwieństwie do wag warstwy Dense, zależą od wejścia?
2. Po co w scaled dot-product attention dzielimy przez sqrt(d_k) i do czego służy maska paddingu?
3. Czym różni się tokenizacja subword HerBERT od słownika słów z laboratorium 5 i co z tego wynika dla słów, których nie było w danych treningowych?
4. Dlaczego transformer potrzebuje informacji o pozycji tokenów, a LSTM nie?
5. Na czym polega fine-tuning modelu pretrenowanego i dlaczego używamy przy nim małego learning rate i niewielu epok?
6. Jakie są koszty HerBERT w porównaniu z TF-IDF + LR (parametry, czas treningu, czas predykcji, sprzęt) i kiedy te koszty się opłacają?

## Zadania rozszerzone

Zadania A i B wykraczają poza podstawowy zakres laboratorium. Korzystają z tokenizera i funkcji z zadań 2 i 3, więc wykonujcie je w tej samej sesji Colaba.

### Zadanie A: Mini-transformer od zera

Zbudujcie mały enkoder transformera w samym Keras i wytrenujcie go od zera na PolEmo, na tych samych tokenach HerBERT, ale z losowo zainicjalizowanymi embeddingami. Trenujcie na całym zbiorze treningowym i walidujcie na całym zbiorze walidacyjnym (gotowa komórka koduje je do `X_train_pelny` i `X_valid_pelny`).

1. Gotowa warstwa `TokenyIPozycje` dodaje embedding tokenu i embedding pozycji (pozycje 0 do `MAX_LEN - 1`).
2. Napiszcie `blok_enkodera(x, maska)`: `MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)` wywołana jako self-attention z `attention_mask=maska[:, None, :]`, dropout, połączenie rezydualne i `LayerNormalization`, potem feed-forward (`Dense(128, activation="relu")`, `Dense(64)`), dropout, znowu połączenie rezydualne i `LayerNormalization`.
3. Napiszcie `zbuduj_mini()`: wejścia jak w HerBERT, `TokenyIPozycje(tokenizer.vocab_size, MAX_LEN, 64)`, maska `keras.ops.cast(attention_mask, "bool")`, jeden blok enkodera, `GlobalAveragePooling1D` wywołana z maską, `Dropout(0.1)`, `Dense(4, activation="softmax")`. Optymalizator `adam`.
4. Wytrenujcie model (`EPOKI_MINI`, `BATCH_MINI`, `EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)`), oceńcie na walidacji i na teście. Porównajcie liczbę parametrów, czas treningu i wynik z HerBERT i z modelami z laboratorium 5.

HerBERT używa tokenu o indeksie 1 jako paddingu, więc `mask_zero=True` w warstwie Embedding nie zadziała; maskę bierzemy z `attention_mask`. Dlaczego transformer bez pretreningu nie wygrywa z modelami z laboratorium 5? Gdzie jest większość jego parametrów? Dlaczego pretrening ma tak duże znaczenie przy 5783 przykładach treningowych?

### Zadanie B: HerBERT jako ekstraktor cech

Zamiast dostrajać 124 mln parametrów, można zamrozić HerBERT, raz policzyć wektory recenzji i wytrenować na nich zwykłą regresję logistyczną. Koszt to tylko inferencja, bez propagacji wstecznej.

1. Wczytajcie świeży, niedostrojony HerBERT (gotowa komórka). Funkcja `wektory_herbert` zwraca dla każdej recenzji wektor tokenu `<s>` i średnią wektorów wszystkich tokenów. Uzupełnijcie w niej uśrednianie z pominięciem pozycji paddingu (pomnóżcie przez `attention_mask` i podzielcie przez liczbę prawdziwych tokenów).
2. Policzcie wektory dla recenzji treningowych (`N_CECHY`, w trybie pełnym wszystkie), `valid_ft` i testu. Zmierzcie czas.
3. Dla obu wariantów (wektor `<s>` i średnia) wytrenujcie `make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))` dla C równego 0,01, 0,1 i 1. Wybierzcie wariant i C według macro F1 na walidacji i dopiero wtedy policzcie wynik na teście.
4. Dla chętnych: test PolEmo 2.0-OUT (gotowa komórka wczytuje 494 recenzje z innych dziedzin). Oceńcie na nim dostrojony HerBERT i regresję na cechach. Policzcie też macro F1 tylko dla klas minus, plus i amb (klasa zero ma w teście OUT jedną recenzję). Porównajcie z wynikami TF-IDF i Embedding + GAP z zadania B laboratorium 5, jeśli je macie.

Dlaczego wektor `<s>` niedostrojonego modelu może być gorszą reprezentacją recenzji niż średnia po tokenach? Ile tracimy względem pełnego fine-tuningu i ile zyskujemy na czasie?

## Co oddać

Notebook `lab06_szablon.ipynb` po uzupełnieniu, z zapisanymi wynikami komórek, wykresami i odpowiedziami na pytania w komórkach „Odpowiedzi”, w formie wskazanej przez prowadzącego. Przed oddaniem uruchomcie notebook od początku do końca w świeżej sesji (Środowisko wykonawcze, Uruchom wszystko) i sprawdźcie, że nie ma błędów. Zostawcie `TRYB_SKROCONY = False`. Wyniki fine-tuningu na GPU mogą się nieznacznie różnić między uruchomieniami, to normalne.

## Źródła

1. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention Is All You Need, NeurIPS 2017, `https://arxiv.org/abs/1706.03762`.
2. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL 2019, `https://aclanthology.org/N19-1423`.
3. R. Mroczkowski, P. Rybak, A. Wróblewska, I. Gawlik, HerBERT: Efficiently Pretrained Transformer-based Language Model for Polish, Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing (BSNLP 2021), `https://aclanthology.org/2021.bsnlp-1.1`. Karta modelu: `https://huggingface.co/allegro/herbert-base-cased` (licencja CC BY 4.0).
4. R. Sennrich, B. Haddow, A. Birch, Neural Machine Translation of Rare Words with Subword Units, ACL 2016, `https://aclanthology.org/P16-1162` (tokenizacja BPE).
5. F. Chollet, Deep Learning with Python, 2nd ed., Manning 2021, rozdział 11 (Deep learning for text), w nim część o architekturze Transformer.
6. I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press 2016, rozdział 10 (Sequence Modeling: Recurrent and Recursive Nets) i podrozdział 12.4 (Natural Language Processing), `https://www.deeplearningbook.org`.
7. J. Kocoń, P. Miłkowski, M. Zaśko-Zielińska, Multi-Level Sentiment Analysis of PolEmo 2.0: Extended Corpus of Multi-Domain Consumer Reviews, CoNLL 2019, `https://aclanthology.org/K19-1092`.
8. P. Rybak, R. Mroczkowski, J. Tracz, I. Gawlik, KLEJ: Comprehensive Benchmark for Polish Language Understanding, ACL 2020, `https://aclanthology.org/2020.acl-main.111`.
9. Zbiory na Hugging Face: `https://huggingface.co/datasets/allegro/klej-polemo2-in` i `https://huggingface.co/datasets/allegro/klej-polemo2-out`; strona benchmarku `https://klejbenchmark.com/tasks/`; korpus pierwotny w repozytorium CLARIN-PL `https://clarin-pl.eu/dspace/handle/11321/710`.
10. Licencja danych: źródła podają różne licencje. Repozytorium CLARIN-PL (korpus pierwotny) podaje CC BY 4.0, strona KLEJ dla wersji w benchmarku CC BY-NC-SA 4.0, a karta zbioru na Hugging Face ma tag CC BY-SA 4.0. Do użytku dydaktycznego wystarcza każda z nich; przy innym użyciu trzeba przyjąć najbardziej restrykcyjną (NC, bez użytku komercyjnego).
11. Dokumentacja Keras: MultiHeadAttention (`https://keras.io/api/layers/attention_layers/multi_head_attention/`), TorchModuleWrapper (`https://keras.io/api/layers/backend_specific_layers/torch_module_wrapper/`), LayerNormalization (`https://keras.io/api/layers/normalization_layers/layer_normalization/`), GlobalAveragePooling1D (`https://keras.io/api/layers/pooling_layers/global_average_pooling1d/`), AdamW (`https://keras.io/api/optimizers/adamw/`).
12. Dokumentacja transformers, strona modelu HerBERT: `https://huggingface.co/docs/transformers/model_doc/herbert`.
