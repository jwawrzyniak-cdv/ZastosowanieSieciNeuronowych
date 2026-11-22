# Laboratorium 5: Sieci neuronowe w przetwarzaniu tekstu

Dane: PolEmo 2.0-IN z benchmarku KLEJ (5783 recenzje treningowe, 723 walidacyjne, 722 testowe, 4 klasy wydźwięku); w zadaniach rozszerzonych dodatkowo test PolEmo 2.0-OUT (494 recenzje z innych dziedzin)
Środowisko: Google Colab z GPU (T4), Keras 3 na backendzie TensorFlow, scikit-learn

## Cel

Tym razem dane nie są obrazami, tylko recenzjami po polsku. Zadanie polega na przypisaniu recenzji jednej z czterech klas: negatywna (minus), pozytywna (plus), ambiwalentna (amb) albo neutralna (zero). Zanim zbudujecie pierwszą sieć, policzycie wynik klasycznego modelu: TF-IDF z regresją logistyczną. To będzie punkt odniesienia, bez którego trudno ocenić, czy sieć w ogóle coś wnosi.

Potem zamienicie tekst na liczby samodzielnie: tokenizer, słownik, kodowanie i padding piszecie w zwykłym Pythonie. Na tych sekwencjach wytrenujecie trzy sieci o różnym podejściu do kolejności słów. Pierwsza uśrednia embeddingi i kolejność ignoruje. Druga, dwukierunkowy LSTM, czyta tekst słowo po słowie. Trzecia, Conv1D, szuka krótkich wzorców w dowolnym miejscu tekstu. Na koniec porównacie wszystkie modele na walidacji, wybierzecie najlepszy i raz sprawdzicie go na teście.

Wynik najlepszej sieci na teście (accuracy i macro F1) zanotujcie. W laboratorium 6 porównacie go z HerBERT, modelem pretrenowanym na dużym korpusie polskich tekstów.

Odpowiedzi na pytania kontrolne możecie dokończyć po zajęciach. Zadania A i B na końcu instrukcji to zadania rozszerzone.

## Przed zajęciami

Przeczytajcie `lab05_wprowadzenie.md`. Opisuje tokenizację, słownik i padding, warstwę Embedding, sieci rekurencyjne i LSTM, Conv1D na tekście oraz macro F1.

Otwórzcie `lab05_szablon.ipynb` w Google Colab i włączcie GPU: Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU. Na CPU wszystko też zadziała, ale LSTM będzie trenował się kilka razy wolniej. Pakiety są w Colabie preinstalowane, nic nie trzeba doinstalowywać.

Komórka importów ustawia `keras.utils.set_random_seed(42)` i wypisuje wersje oraz informację o GPU. Wszystkie parametry, od których zależy czas treningu (rozmiar słownika, `MAX_LEN`, wymiar embeddingu, liczba epok), są w jednej komórce konfiguracyjnej. Wczytanie danych jest gotowe: trzy pliki CSV z Hugging Face (`allegro/klej-polemo2-in`), kolumny `sentence` (tekst) i `target` (etykieta), wczytywane przez `pd.read_csv(url)`. Gotowe są też funkcje pomocnicze `ocen` (accuracy i macro F1, zapis do słownika `wyniki`), `tabela_wynikow`, `przewidz`, `rysuj_historie` i `rysuj_macierz`.

Teksty w PolEmo są już podzielone na tokeny spacjami: przecinki i kropki stoją osobno („człowiek przez duże C . Bardzo”). Część recenzji jest napisana bez polskich znaków. Nie poprawiajcie tego, to część zadania.

## Zadanie 1: Dane

1. Uruchomcie gotową komórkę wczytującą dane. Sprawdźcie, jak etykiety `__label__meta_minus_m`, `__label__meta_plus_m`, `__label__meta_amb` i `__label__meta_zero` zostały zamienione na nazwy minus, plus, amb i zero.
2. Policzcie rozkład klas w train, valid i test (liczności i procenty).
3. Narysujcie histogram długości recenzji w słowach (`str.split()`). Podajcie medianę, 90. i 95. percentyl. Porównajcie medianę długości w poszczególnych klasach.
4. Wypiszcie po dwie recenzje z każdej klasy (pierwsze 300 znaków).

Czy zbiór jest niezbalansowany? Jaką accuracy dałby model, który zawsze odpowiada najliczniejszą klasą? Dlaczego oprócz accuracy będziemy liczyć macro F1? Czym różnią się teksty z klasy zero od pozostałych?

## Zadanie 2: Baseline klasyczny, TF-IDF i regresja logistyczna

1. Zbudujcie `TfidfVectorizer` z unigramami i bigramami (`ngram_range=(1, 2)`), `min_df=2` i `sublinear_tf=True`. Dopasujcie go wyłącznie na zbiorze treningowym, a valid i test tylko przekształćcie.
2. Wytrenujcie `LogisticRegression(C=10, max_iter=2000)` i oceńcie ją na walidacji funkcją `ocen("TF-IDF + LR", "valid", ...)`.
3. Wypiszcie 10 n-gramów o największych wagach dla każdej klasy (`logreg.coef_`, `tfidf.get_feature_names_out()`).

Dlaczego przy niecałych 6 tys. recenzji TF-IDF z regresją logistyczną jest tak trudny do pobicia? Które n-gramy charakteryzują klasę amb, a które klasę zero, i co to mówi o tych klasach?

## Zadanie 3: Tokenizacja i słownik

Keras ma warstwę `keras.layers.TextVectorization`, która robi tokenizację, słownik i padding w jednym kroku. Działa jednak tylko z backendem TensorFlow, a w laboratorium 6 przejdziemy na backend torch. Dlatego tokenizer i słownik napiszecie sami, w zwykłym Pythonie. Przy backendzie TensorFlow `TextVectorization` jest wygodną alternatywą.

1. Napiszcie `tokenizuj(tekst)`: zamiana na małe litery i wyciągnięcie wyrazów wyrażeniem regularnym `[a-ząćęłńóśźż0-9]+`. Interpunkcja znika.
2. Zbudujcie słownik z `MAX_TOKENS - 2` (czyli 19998) najczęstszych tokenów zbioru treningowego (`collections.Counter`). Indeks 0 jest zarezerwowany dla paddingu, indeks 1 dla tokenu spoza słownika (OOV), słowa dostają indeksy od 2. Przygotujcie też słownik odwrotny `indeks_do_slowa`.
3. Napiszcie `koduj(tekst)`, która zwraca listę indeksów. Policzcie odsetek tokenów OOV na walidacji.
4. Narysujcie histogram długości recenzji w tokenach, podajcie percentyle (50, 90, 95, 99) i odsetek recenzji dłuższych niż `MAX_LEN = 256`.
5. Zamieńcie teksty na macierze `X_train`, `X_valid`, `X_test` funkcją `keras.utils.pad_sequences(..., maxlen=MAX_LEN, padding="post", truncating="post")`. Sprawdźcie, czy nie ma wierszy złożonych z samego paddingu. Na przykładzie `[[5, 6, 7], [5, 6, 7, 8, 9, 10, 11]]` z `maxlen=5` pokażcie wszystkie cztery kombinacje `padding` i `truncating` (`"pre"` i `"post"`).

Skąd tak duży odsetek OOV przy słowniku 20 tys. słów (pomyślcie o polskiej fleksji)? Czy `MAX_LEN = 256` to rozsądny kompromis? Kiedy lepiej obcinać początek recenzji (`truncating="pre"`), a kiedy koniec? Dlaczego słownik budujemy tylko na zbiorze treningowym?

## Zadanie 4: Embedding i GlobalAveragePooling1D

1. Napiszcie funkcję `trenuj(model)`: kompilacja (optymalizator `adam`, strata `sparse_categorical_crossentropy`, metryka accuracy), `fit` na `X_train` z walidacją na `X_valid`, `epochs=EPOCHS`, `batch_size=BATCH_SIZE`, `verbose=2` i callback `EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)`. Niech wypisuje czas treningu i numer najlepszej epoki i zwraca historię.
2. Zbudujcie model „worek embeddingów”: `keras.Input(shape=(MAX_LEN,), dtype="int32")`, `Embedding(MAX_TOKENS, EMB_DIM, mask_zero=True)`, `GlobalAveragePooling1D`, `Dropout(0.3)`, `Dense(4, activation="softmax")`. Wyświetlcie `summary()`. Gdzie jest prawie cała liczba parametrów?
3. Wytrenujcie model, narysujcie historię (`rysuj_historie`) i oceńcie go na walidacji (`ocen("Embedding + GAP", "valid", ...)`). Model zapiszcie w słowniku `modele["Embedding + GAP"]`.
4. Wytrenujcie ten sam model z `mask_zero=False` i porównajcie wynik i liczbę epok.

`mask_zero=True` sprawia, że warstwa Embedding przekazuje dalej maskę: pozycje z indeksem 0 są pomijane przez warstwy, które maskę obsługują (GlobalAveragePooling1D, LSTM). Co bez maski trafia do średniej w krótkiej recenzji? Co tracimy, uśredniając embeddingi słów? Jak taki model poradzi sobie z „nie polecam” w porównaniu z „polecam”?

## Zadanie 5: Bidirectional LSTM i Conv1D

1. Zbudujcie BiLSTM: `Embedding(MAX_TOKENS, EMB_DIM, mask_zero=True)`, `Bidirectional(LSTM(LSTM_UNITS))`, `Dropout(0.5)`, `Dense(4, activation="softmax")`. Zostawcie domyślne parametry LSTM. W Keras 3 na backendzie TensorFlow warstwa LSTM użyje na GPU szybkiej implementacji cuDNN tylko wtedy, gdy `activation="tanh"`, `recurrent_activation="sigmoid"`, `recurrent_dropout=0`, `unroll=False` i `use_bias=True`, a przy masce sekwencje są dopełnione zerami na końcu (`padding="post"`) i żadna nie składa się z samego paddingu. Dlatego nie używamy `recurrent_dropout`, a dropout stoi za warstwą rekurencyjną. Z maską i `padding="pre"` Keras na GPU nie przełącza się po cichu na wolniejszą implementację, tylko przerywa obliczenia błędem o masce, która nie jest dopełniona z prawej strony.
2. Zbudujcie Conv1D: `Embedding(MAX_TOKENS, EMB_DIM)` (bez maski, Conv1D jej nie obsługuje), `Conv1D(128, 5, activation="relu")`, `GlobalMaxPooling1D`, `Dropout(0.5)`, `Dense(4, activation="softmax")`.
3. Wytrenujcie oba modele funkcją `trenuj` (przed każdym `keras.utils.set_random_seed(42)`), narysujcie historie i oceńcie na walidacji. Zapiszcie je w `modele["BiLSTM"]` i `modele["Conv1D"]`.
4. Porównajcie liczbę parametrów, czas jednej epoki, numer najlepszej epoki i wyniki trzech sieci.

Po co LSTM dwukierunkowy, skoro recenzję i tak czytamy od początku? Co „widzi” jeden filtr Conv1D o szerokości 5 i co robi z jego odpowiedziami GlobalMaxPooling1D? Która sieć najszybciej się przeucza i po czym to widać na wykresach?

## Zadanie 6: Porównanie modeli i test

1. Wyświetlcie tabelę wyników na walidacji (`tabela_wynikow("valid")`) i wybierzcie najlepszą sieć według macro F1 na walidacji.
2. Dopiero teraz, jeden raz, policzcie wyniki na zbiorze testowym dla TF-IDF + LR i trzech sieci. Zestawcie accuracy i macro F1 na valid i test w jednej tabeli. Po spojrzeniu na test niczego już nie zmieniajcie.
3. Dla najlepszej sieci wypiszcie `classification_report` na teście i narysujcie macierz pomyłek, obok macierz dla TF-IDF + LR.
4. Zanotujcie accuracy i macro F1 najlepszej sieci na teście (oraz wynik TF-IDF + LR). Te liczby wpiszecie w laboratorium 6 do porównania z HerBERT.

Która klasa jest najtrudniejsza i z którymi klasami jest mylona? Czy ranking modeli na teście jest taki sam jak na walidacji? Jak duża różnica w accuracy odpowiada jednej recenzji przy 723 przykładach i co z tego wynika dla wyboru modelu?

## Pytania kontrolne

1. Dlaczego słownik i TF-IDF dopasowujemy tylko na zbiorze treningowym?
2. Co oznaczają indeksy 0 i 1 w naszym słowniku i co robi `mask_zero=True` w warstwie Embedding?
3. Dlaczego warstwa Embedding ma tak dużo parametrów i co to oznacza przy kilku tysiącach przykładów treningowych?
4. Jakie warunki musi spełnić warstwa LSTM w Keras 3, żeby na GPU użyć cuDNN, i co to ma wspólnego z `padding="post"`?
5. Czym różni się GlobalAveragePooling1D od GlobalMaxPooling1D w zastosowaniu do tekstu?
6. Dlaczego przy niezbalansowanych klasach accuracy i macro F1 mogą prowadzić do różnych wniosków i dlaczego zbiór testowy sprawdzamy tylko raz?

## Zadania rozszerzone

Zadania A i B wykraczają poza podstawowy zakres laboratorium. Oba korzystają z modeli wytrenowanych w zadaniach 4 do 6, więc wykonujcie je w tej samej sesji Colaba.

### Zadanie A: Wyuczone embeddingi

1. Wyciągnijcie macierz wag warstwy Embedding z modelu Embedding + GAP i z BiLSTM (`keras.ops.convert_to_numpy(warstwa.embeddings)`).
2. Napiszcie `sasiedzi(slowo, E, k=8)`, która zwraca k najbliższych słów według podobieństwa cosinusowego. Szukajcie tylko wśród 5000 najczęstszych słów (indeksy 2 do 5001), bo rzadkie słowa mają słabo wyuczone wektory.
3. Wypiszcie sąsiadów słów „polecam”, „brud”, „lekarz”, „personel” i „drogo” w obu modelach.
4. W modelu Embedding + GAP wynik klasy dla tekstu złożonego z jednego słowa to iloczyn jego embeddingu z wagami warstwy Dense plus bias. Policzcie te „wkłady” dla 5000 najczęstszych słów i wypiszcie po 12 słów najmocniej wskazujących na każdą klasę (względem średniej z czterech klas).

Czy sąsiedzi przypominają synonimy, jak w word2vec? Czego nauczyły się te embeddingi i dlaczego właśnie tego? Czym różnią się sąsiedzi w modelu GAP i BiLSTM?

### Zadanie B: Zmiana domeny, PolEmo 2.0-OUT

Benchmark KLEJ zawiera drugą wersję zadania, PolEmo 2.0-OUT. Zbiór treningowy jest identyczny (medycyna i hotele), ale test pochodzi z innych dziedzin: recenzje produktów i opinie o zajęciach na uczelni. Plik: `https://huggingface.co/datasets/allegro/klej-polemo2-out/resolve/main/test.csv` (494 recenzje, wczytanie w gotowej komórce).

1. Porównajcie rozkład klas w teście IN i OUT. Policzcie odsetek tokenów OOV na teście OUT (słownik z zadania 3) i porównajcie z walidacją IN.
2. Oceńcie TF-IDF + LR i najlepszą sieć z zadania 6 na teście OUT (`ocen(nazwa, "test OUT", ...)`). Bez ponownego trenowania.
3. Klasa zero ma w teście OUT tylko jedną recenzję, więc macro F1 po czterech klasach jest mało wiarygodne. Policzcie także macro F1 tylko dla minus, plus i amb (`f1_score(..., labels=[0, 1, 2], average="macro")`) oraz recall każdej klasy w obu domenach.
4. Policzcie accuracy, jakiej należałoby się spodziewać na OUT, gdyby recall każdej klasy był taki jak na teście IN, a zmienił się tylko rozkład klas. Narysujcie macierze pomyłek na OUT.

Jaką część spadku accuracy tłumaczy sama zmiana rozkładu klas, a jaką inne słownictwo? Który model traci więcej i dlaczego? Czego to uczy o raportowaniu wyników modelu wytrenowanego na jednej domenie?

## Co oddać

Notebook `lab05_szablon.ipynb` po uzupełnieniu, z zapisanymi wynikami komórek, wykresami i odpowiedziami na pytania w komórkach „Odpowiedzi”, w formie wskazanej przez prowadzącego. Przed oddaniem uruchomcie notebook od początku do końca (Środowisko wykonawcze, Uruchom wszystko) i sprawdźcie, że nie ma błędów. Wyniki na GPU mogą się nieznacznie różnić między uruchomieniami, to normalne.

Zanotujcie poza notebookiem accuracy i macro F1 najlepszej sieci na teście PolEmo 2.0-IN oraz wynik TF-IDF + LR. Będą potrzebne w laboratorium 6.

## Źródła

1. F. Chollet, Deep Learning with Python, 2nd ed., Manning 2021, rozdział 11 (Deep learning for text).
2. I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press 2016, rozdział 10 (Sequence Modeling: Recurrent and Recursive Nets) i podrozdział 12.4 (Natural Language Processing), `https://www.deeplearningbook.org`.
3. S. Hochreiter, J. Schmidhuber, Long Short-Term Memory, Neural Computation 9(8), 1997.
4. Y. Kim, Convolutional Neural Networks for Sentence Classification, EMNLP 2014.
5. J. Kocoń, P. Miłkowski, M. Zaśko-Zielińska, Multi-Level Sentiment Analysis of PolEmo 2.0: Extended Corpus of Multi-Domain Consumer Reviews, CoNLL 2019, `https://aclanthology.org/K19-1092`.
6. P. Rybak, R. Mroczkowski, J. Tracz, I. Gawlik, KLEJ: Comprehensive Benchmark for Polish Language Understanding, ACL 2020, `https://aclanthology.org/2020.acl-main.111`.
7. Zbiory na Hugging Face: `https://huggingface.co/datasets/allegro/klej-polemo2-in` i `https://huggingface.co/datasets/allegro/klej-polemo2-out`; strona benchmarku `https://klejbenchmark.com/tasks/`; korpus pierwotny w repozytorium CLARIN-PL `https://clarin-pl.eu/dspace/handle/11321/710`.
8. Licencja danych: źródła podają różne licencje. Repozytorium CLARIN-PL (korpus pierwotny) podaje CC BY 4.0, strona KLEJ dla wersji w benchmarku CC BY-NC-SA 4.0, a karta zbioru na Hugging Face ma tag CC BY-SA 4.0. Do użytku dydaktycznego wystarcza każda z nich; przy innym użyciu trzeba przyjąć najbardziej restrykcyjną (NC, bez użytku komercyjnego).
9. Dokumentacja Keras: warstwy Embedding (`https://keras.io/api/layers/core_layers/embedding/`), LSTM (`https://keras.io/api/layers/recurrent_layers/lstm/`), Bidirectional (`https://keras.io/api/layers/recurrent_layers/bidirectional/`), Conv1D (`https://keras.io/api/layers/convolution_layers/convolution1d/`), TextVectorization (`https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/`).
10. Dokumentacja scikit-learn: `TfidfVectorizer` (`https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html`) i `LogisticRegression` (`https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html`).
