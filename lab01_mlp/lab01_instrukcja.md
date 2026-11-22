# Laboratorium 1: Perceptron wielowarstwowy i pętla treningowa

Dane: Fashion-MNIST (`keras.datasets.fashion_mnist`), 60 000 obrazów treningowych i 10 000 testowych 28x28 w skali szarości, 10 klas
Środowisko: Google Colab z GPU (T4), Keras 3 na backendzie TensorFlow, scikit-learn, pandas, matplotlib

## Cel

Na tych zajęciach budujecie pierwszą sieć neuronową w Keras i sprawdzacie, co dzieje się w czasie jej treningu. Zaczynacie od regresji softmax, czyli sieci bez warstw ukrytych, która wyznacza punkt odniesienia. Potem dokładacie dwie warstwy ukryte i otrzymujecie perceptron wielowarstwowy (MLP). Liczycie jego parametry, czytacie krzywe uczenia i decydujecie, kiedy trening przestaje pomagać.

Drugą część zajmuje sama pętla treningowa. Ten sam MLP trenujecie najpierw przez `fit`, a potem własną pętlą z `tf.GradientTape`, w której forward pass, liczenie gradientów i aktualizację wag zapisujecie jawnie. Dalej sprawdzacie w krótkich przebiegach, jak na trening wpływają learning rate, rozmiar batcha i funkcja aktywacji. Na końcu, jeden raz, oceniacie model na zbiorze testowym i oglądacie, które klasy odzieży myli i z jaką pewnością.

Notebook z tego laboratorium (funkcje `zbuduj_mlp`, `rysuj_krzywe`, analiza błędów) przyda się w laboratorium 2, gdzie ten sam typ danych obrazowych trafi do sieci konwolucyjnej.

Zadania 8 do 11 na końcu instrukcji to zadania rozszerzone o inicjalizacji wag, BatchNormalization i zanikającym gradiencie.

## Przed zajęciami

Przeczytajcie `lab01_wprowadzenie.md`. Zebrane tam pojęcia (funkcja straty, spadek gradientu, propagacja wsteczna, funkcje aktywacji, przeuczenie, inicjalizacja wag) są potrzebne we wszystkich zadaniach.

Otwórzcie `lab01_szablon.ipynb` w Google Colab i włączcie GPU: Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU. Wszystkie biblioteki są w Colabie zainstalowane, więc notebook nie ma komórki instalacyjnej. Uruchomcie pierwsze komórki: importy wypisują wersję Keras, nazwę backendu (ma być `tensorflow`) i listę dostępnych GPU. Zbiór Fashion-MNIST pobiera się sam przy pierwszym wywołaniu `load_data()`.

Komórka z parametrami na początku notebooka (`EPOKI_BASELINE`, `EPOKI_MLP`, `EPOKI_PETLA`, `EPOKI_EKSP`, `BATCH`, `UDZIAL_WALIDACJI`) ustala liczbę epok we wszystkich zadaniach. Nie zmieniajcie jej, chyba że prowadzący powie inaczej: przebiegi są krótkie po to, żeby zmieściły się w zajęciach.

Gotowe w szablonie są: wczytanie danych i lista nazw klas, funkcje rysujące (`pokaz_obrazy`, `rysuj_krzywe`, `rysuj_porownanie`, `rysuj_macierz_pomylek`) oraz do zadań rozszerzonych funkcje pomocnicze `nowa_inicjalizacja`, `odchylenia_aktywacji` i `zbuduj_gleboka`. Resztę piszecie sami w komórkach oznaczonych `# TODO`. Każdy model budujcie po wywołaniu `keras.utils.set_random_seed(RANDOM_STATE)`, wtedy porównania między przebiegami są uczciwsze.

## Zadanie 1: Dane

Etykiety to liczby od 0 do 9. Lista `KLASY` w szablonie zawiera polskie nazwy w tej samej kolejności: koszulka (T-shirt/top), spodnie (Trouser), sweter (Pullover), sukienka (Dress), płaszcz (Coat), sandał (Sandal), koszula (Shirt), but sportowy (Sneaker), torba (Bag), botek (Ankle boot).

1. Wypiszcie kształty i typy tablic `x_train_full`, `y_train_full`, `x_test`, `y_test` oraz najmniejszą i największą wartość piksela.
2. Policzcie przykłady każdej klasy w zbiorze treningowym i testowym (`np.bincount`) i pokażcie je w tabeli z nazwami klas.
3. Narysujcie funkcją `pokaz_obrazy` po dwa losowe przykłady z każdej klasy z podpisami z listy `KLASY`.
4. Zamieńcie piksele na `float32` w zakresie [0, 1]. Wydzielcie z treningu 10% przykładów na walidację funkcją `train_test_split` z `stratify=y_train_full` i `random_state=RANDOM_STATE`. Zbiór testowy też znormalizujcie (`x_test_n`), ale nie używajcie go aż do zadania 6.

Dlaczego dzielimy przez 255 i co by się zmieniło, gdyby sieć dostawała wartości od 0 do 255? Po co wydzielać walidację z treningu, skoro zbiór testowy jest gotowy? Czy klasy są zrównoważone i co z tego wynika dla dokładności jako miary jakości?

## Zadanie 2: Baseline: regresja softmax

1. Zbudujcie `model_baseline`: `keras.Input(shape=(28, 28))`, `Flatten`, `Dense(10, activation="softmax")`. Skompilujcie go z `optimizer=keras.optimizers.Adam(learning_rate=1e-3)`, `loss="sparse_categorical_crossentropy"` i `metrics=["accuracy"]`. W `compile` podawajcie argumenty z nazwami. Wypiszcie `summary()`.
2. Przed treningiem policzcie `evaluate` na walidacji i porównajcie stratę z `np.log(10)`.
3. Wytrenujcie model przez `EPOKI_BASELINE` epok z `validation_data=(x_val, y_val)`, `batch_size=BATCH` i `verbose=2`. Zmierzcie czas.
4. Narysujcie krzywe uczenia (`rysuj_krzywe`) i zapiszcie dokładność walidacyjną z ostatniej epoki jako `acc_baseline`.

Ile parametrów ma model i skąd ta liczba? Dlaczego strata przed treningiem jest bliska ln 10? Czy krzywe treningowa i walidacyjna się rozjeżdżają i co to mówi o pojemności tego modelu?

## Zadanie 3: Perceptron wielowarstwowy

1. Napiszcie funkcję `zbuduj_mlp(aktywacja="relu", nazwa="mlp")`, która zwraca nieskompilowany model: `Input((28, 28))`, `Flatten`, `Dense(256)`, `Dense(128)` (obie z aktywacją z argumentu), `Dense(10, activation="softmax")`. Użyjecie jej w zadaniach 4 i 5.
2. Zbudujcie `model_mlp`, skompilujcie go jak baseline, wypiszcie `summary()` i wytrenujcie przez `EPOKI_MLP` epok.
3. Policzcie ręcznie liczbę parametrów pierwszej warstwy ukrytej (wagi i biasy) oraz całego modelu i porównajcie z `count_params()`. Lista `model_mlp.layers` nie zawiera warstwy `Input`, więc pierwszy element to `Flatten`.
4. Narysujcie krzywe uczenia. Znajdźcie epokę z najlepszą `val_accuracy` i epokę z najmniejszą `val_loss`. Ile zyskaliście względem baseline?

Od której epoki krzywe straty treningowej i walidacyjnej się rozjeżdżają i co to oznacza? Jak to możliwe, że dokładność walidacyjna jeszcze nie spada, choć strata walidacyjna już rośnie? Którą epokę wybralibyście na model końcowy? Co by się stało, gdyby warstwy ukryte nie miały funkcji aktywacji?

## Zadanie 4: Własna pętla treningowa z `tf.GradientTape`

`fit` w każdym kroku przepuszcza batch przez model, liczy stratę, wyznacza gradienty, aktualizuje wagi i odświeża metryki. Tutaj zapiszecie te kroki sami. Ten kod działa tylko na backendzie TensorFlow: gradienty liczy `tf.GradientTape`, dane podaje `tf.data`, a `tf.function` kompiluje krok do grafu. Na backendzie JAX czy PyTorch pętla wygląda inaczej (odpowiednio `jax.value_and_grad` i `loss.backward()`). Model, optymalizator, strata i metryki pochodzą z Keras i są wspólne.

1. Przygotujcie `ds_train` (`tf.data.Dataset.from_tensor_slices`, `shuffle(len(x_train), seed=RANDOM_STATE)`, `batch(BATCH)`) i `ds_val` (bez tasowania, batch 1024).
2. Po ustawieniu ziarna zbudujcie `model_petla = zbuduj_mlp(nazwa="mlp_petla")`, optymalizator `keras.optimizers.Adam(learning_rate=1e-3)`, funkcję straty `keras.losses.SparseCategoricalCrossentropy()` i metryki: `keras.metrics.Mean()` dla średniej straty oraz dwie `keras.metrics.SparseCategoricalAccuracy()` (trening, walidacja).
3. Napiszcie `krok_treningowy(x, y)`: w bloku `with tf.GradientTape() as tasma:` predykcja z `training=True` i strata, potem `tasma.gradient(strata, model_petla.trainable_weights)`, `optymalizator.apply_gradients(zip(gradienty, model_petla.trainable_weights))` i aktualizacja metryk. Napiszcie `krok_walidacyjny(x, y)` z `training=False`. Gotowa linia w szablonie opakowuje obie funkcje w `tf.function`.
4. Napiszcie pętlę po `EPOKI_PETLA` epokach. Na początku każdej epoki wyzerujcie metryki (`reset_state()`), po epoce wypiszcie stratę, dokładność treningową, dokładność walidacyjną i czas.
5. Wytrenujcie ten sam model (ziarno, `zbuduj_mlp`, te same ustawienia) przez `fit` przez `EPOKI_PETLA` epok. Zestawcie w tabeli stratę i dokładność walidacyjną z obu sposobów oraz łączny czas.
6. Zmierzcie czas 100 kroków na tych samych batchach (`list(ds_train.take(100))`) z `krok_treningowy_tf` i z `krok_treningowy` bez `tf.function`.

Które linie pętli odpowiadają za forward pass, backpropagation i aktualizację wag? Dlaczego wyniki pętli i `fit` nie są identyczne, choć architektura, ziarno i hiperparametry są te same? Po co `training=True`, skoro ten MLP działa tak samo w obu trybach? Skąd różnica czasu z `tf.function` i bez niego, i kiedy własna pętla jest potrzebna, a kiedy wystarczy `fit`?

## Zadanie 5: Eksperymenty: learning rate, batch, aktywacja

Konfiguracja bazowa: Adam, learning rate 1e-3, batch 128, ReLU. Każdy przebieg trwa `EPOKI_EKSP` (5) epok i zaczyna się od `keras.utils.set_random_seed(RANDOM_STATE)`. Zmieniacie jeden parametr naraz. W `fit` ustawcie `verbose=0`, żeby notebook nie urósł.

1. Napiszcie `eksperyment(lr=1e-3, batch=BATCH, aktywacja="relu")`, która buduje model funkcją `zbuduj_mlp`, kompiluje go z Adamem, trenuje i zwraca historię oraz czas treningu.
2. Wykonajcie przebiegi: learning rate 1e-4, 1e-3, 1e-2, 1e-1; batch 32, 128, 256; aktywacja relu, sigmoid, tanh. Konfigurację bazową liczcie raz i użyjcie jej w każdej z trzech grup (razem 8 przebiegów). Zbierzcie wyniki w tabeli: grupa, wartość, liczba kroków na epokę, `val_accuracy` w ostatniej epoce, najlepsza `val_accuracy`, strata treningowa i walidacyjna, czas.
3. Narysujcie trzy panele (`plt.subplots(1, 3)`), w każdym `rysuj_porownanie` dla jednej grupy.

Co się dzieje przy learning rate 1e-1, a co przy 1e-4, i czego spodziewacie się po tych przebiegach po 50 epokach? Dlaczego sigmoid w warstwach ukrytych uczy się wolniej niż ReLU i tanh? Ile aktualizacji wag wykonują warianty z batchem 32 i 256 w tych samych epokach i czy porównanie przy tej samej liczbie epok jest uczciwe? Czy różnicę 0,3 punktu procentowego między dwoma przebiegami uznalibyście za istotną?

## Zadanie 6: Ocena na zbiorze testowym i analiza błędów

Modelem końcowym jest MLP z zadania 3. Zbiór testowy wykorzystujecie w tym zadaniu pierwszy i jedyny raz.

1. Policzcie stratę i dokładność `model_mlp` na `x_test_n` i porównajcie je z wynikiem walidacyjnym z ostatniej epoki zadania 3.
2. Wyznaczcie prawdopodobieństwa (`predict`), przewidziane klasy (`argmax`) i macierz pomyłek (`sklearn.metrics.confusion_matrix`). Narysujcie ją funkcją `rysuj_macierz_pomylek`.
3. Policzcie trafność w każdej klasie (przekątna podzielona przez sumę wiersza) i wypiszcie sześć najczęstszych par (klasa prawdziwa, klasa przewidziana).
4. Wśród błędnych predykcji znajdźcie 12, przy których model był najbardziej pewny, i narysujcie je z podpisem: klasa prawdziwa, klasa przewidziana, prawdopodobieństwo. Ile błędów ma pewność powyżej 0,99?

Dlaczego zbiór testowy wykorzystujemy raz, na końcu? Które klasy model myli najczęściej i dlaczego akurat te? Czy któryś z najbardziej pewnych błędów wygląda raczej na wątpliwą etykietę niż na błąd modelu i co oznacza pewność 0,99 przy błędnej odpowiedzi? Co w budowie MLP utrudnia odróżnienie koszuli od koszulki i swetra?

## Zadanie 7: Pytania kontrolne

Odpowiedzcie w notebooku na pytania z następnej sekcji, po jednym do trzech zdań na pytanie.

## Pytania kontrolne

1. Czym różni się regresja softmax od MLP i dlaczego MLP bez nieliniowych funkcji aktywacji nie jest od niej silniejszy?
2. Ile parametrów ma warstwa `Dense` z n wejściami i m neuronami?
3. Wymieńcie kroki jednej iteracji pętli treningowej. Które z nich wykonuje `tf.GradientTape`, a które optymalizator?
4. Po czym na krzywych uczenia poznacie przeuczenie, a po czym zbyt duży learning rate?
5. Dlaczego hiperparametry dobieramy na walidacji, a zbioru testowego używamy raz, na końcu?
6. Dlaczego sigmoid w warstwach ukrytych uczy się wolniej niż ReLU?
7. Czym różni się epoka od kroku (batcha)? Ile kroków ma epoka przy 54 000 przykładach i batchu 128?

## Zadania rozszerzone

Zadania 8 do 11 wykraczają poza podstawowy zakres laboratorium. W notebooku są na końcu, w sekcji "Zadania rozszerzone". Wszystkie przebiegi trwają `EPOKI_EKSP` epok.

### Zadanie 8: Inicjalizacja wag

1. Napiszcie `zbuduj_mlp_rozsz(inicjalizacja="glorot_uniform", bn=False, aktywacja="relu")`. Bez BN to ten sam MLP co w zadaniu 3, ale każda warstwa `Dense` dostaje `kernel_initializer=nowa_inicjalizacja(inicjalizacja)`. Gotowa funkcja `nowa_inicjalizacja` przyjmuje `"glorot_uniform"`, `"he_normal"` albo `"normal_0.001"` (rozkład normalny o odchyleniu 0,001) i dla każdej warstwy tworzy nowy obiekt. Wariant z BN opisuje zadanie 9.
2. Dla trzech inicjalizacji policzcie gotową funkcją `odchylenia_aktywacji` odchylenie standardowe wyjść warstw na 256 obrazach walidacyjnych, przed treningiem.
3. Wytrenujcie MLP z każdą inicjalizacją optymalizatorem `keras.optimizers.SGD(learning_rate=0.01)` (bez momentum), a wariant `"normal_0.001"` dodatkowo Adamem z learning rate 1e-3. Zapiszcie stratę przed treningiem, po pierwszej epoce i dokładność walidacyjną na końcu.

Co dzieje się z sygnałem w sieci przy wagach o odchyleniu 0,001? Dlaczego SGD przy takich wagach praktycznie stoi w miejscu, a Adam się uczy? Czy różnica między inicjalizacją Glorota i He ma w tej sieci znaczenie?

### Zadanie 9: BatchNormalization

1. Uzupełnijcie `zbuduj_mlp_rozsz` o wariant `bn=True`: każda warstwa ukryta to `Dense(n, use_bias=False)`, `BatchNormalization()`, `Activation(aktywacja)`. Wypiszcie `summary()` i policzcie wagi trenowalne i nietrenowalne.
2. Wytrenujcie model z BN przy wagach `"normal_0.001"` i SGD oraz przy domyślnej inicjalizacji i Adamie. Porównajcie z odpowiednikami bez BN (zadanie 8 i konfiguracja bazowa z zadania 5).

Ile parametrów dodaje BatchNormalization i które z nich nie są trenowane przez optymalizator? Dlaczego BN ratuje trening przy zbyt małych wagach? Czy poprawił wynik konfiguracji bazowej i jak oceniacie wiarygodność tej różnicy?

### Zadanie 10: Zanikający gradient

Gotowa funkcja `zbuduj_gleboka(aktywacja, liczba_warstw=10, szerokosc=64, bn=False)` buduje MLP z dziesięcioma warstwami ukrytymi po 64 neurony.

1. Napiszcie `normy_gradientow(model, x, y)`: w bloku `tf.GradientTape` policzcie średnią stratę na jednym batchu (z `training=True`), wyznaczcie gradienty po jądrach (`kernel`) wszystkich warstw `Dense` i zwróćcie listę ich norm (`tf.norm`). To również kod dla backendu TensorFlow.
2. Policzcie normy przed treningiem na batchu 256 obrazów treningowych dla sieci z aktywacją sigmoid, tanh i relu oraz z sigmoidą i BatchNormalization. Pokażcie je w tabeli i na wykresie z osią pionową w skali logarytmicznej (`ax.semilogy`). Dla każdego wariantu podajcie stosunek normy w warstwie wyjściowej do normy w pierwszej warstwie.
3. Wytrenujcie warianty sigmoid, relu i sigmoid z BN (Adam, learning rate 1e-3) i porównajcie dokładność walidacyjną.

Ile razy norma gradientu w warstwie wyjściowej jest większa niż w pierwszej przy sigmoidzie, a ile przy ReLU? Jak ten pomiar przekłada się na trening głębokiej sieci z sigmoidą i dlaczego nie pomaga nawet Adam? Dlaczego tanh wypada tu znacznie lepiej niż sigmoid i co zmienia BatchNormalization?

### Zadanie 11: Wnioski

Odpowiedzcie w notebooku na pytania z zadań 8 do 10 i dopiszcie dwa, trzy zdania o tym, który z trzech mechanizmów (inicjalizacja, BatchNormalization, wybór aktywacji) zastosowalibyście najpierw w nowej, głębokiej sieci i dlaczego.

## Co oddać

Notebook `lab01_szablon.ipynb` po uzupełnieniu, z zapisanymi wynikami komórek, wykresami i odpowiedziami na pytania po każdym zadaniu, w formie wskazanej przez prowadzącego. Przed oddaniem uruchomcie go od początku do końca (Środowisko wykonawcze, Uruchom wszystko) i sprawdźcie, że nie ma błędów. Jeżeli zmieniliście parametry z komórki konfiguracyjnej, napiszcie to w pierwszej komórce notebooka. Jeśli robicie zadania rozszerzone, oddajecie ten sam notebook z uzupełnioną sekcją "Zadania rozszerzone".

## Źródła

1. F. Chollet, Deep Learning with Python, 2nd ed., Manning 2021: rozdział 2 (matematyczne podstawy sieci, gradient i `GradientTape`), rozdział 3 (wprowadzenie do Keras i TensorFlow), rozdział 4 (klasyfikacja), rozdział 5 (podstawy uczenia maszynowego: walidacja, przeuczenie), rozdział 7 (Keras w szczegółach, własne pętle treningowe).
2. I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press 2016 (www.deeplearningbook.org): rozdział 6 (Deep Feedforward Networks), rozdział 8 (Optimization for Training Deep Models, w tym inicjalizacja parametrów i batch normalization).
3. Dokumentacja Keras: zbiór Fashion-MNIST, https://keras.io/api/datasets/fashion_mnist/ ; przewodnik "Writing a training loop from scratch in TensorFlow", https://keras.io/guides/writing_a_custom_training_loop_in_tensorflow/ .
4. H. Xiao, K. Rasul, R. Vollgraf, Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms, arXiv:1708.07747, 2017. Zbiór jest udostępniony na licencji MIT.
5. X. Glorot, Y. Bengio, Understanding the difficulty of training deep feedforward neural networks, AISTATS 2010.
6. K. He, X. Zhang, S. Ren, J. Sun, Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, ICCV 2015.
7. S. Ioffe, C. Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, ICML 2015.
8. D. P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization, ICLR 2015 (arXiv:1412.6980).
