# Laboratorium 2: Sieci splotowe w klasyfikacji obrazów

Dane: CIFAR-10 (`keras.datasets.cifar10`): 50 000 obrazów treningowych i 10 000 testowych, kolorowe, 32x32 piksele, 10 klas
Środowisko: Google Colab z GPU (T4), Keras 3 na backendzie TensorFlow, numpy, scipy, pandas, matplotlib (wszystko preinstalowane w Colabie)

## Cel

Na pierwszym laboratorium MLP klasyfikował Fashion-MNIST, gdzie obrazy są małe, jednokanałowe i dobrze wycentrowane. CIFAR-10 jest trudniejszy: zdjęcia są kolorowe, obiekty leżą w różnych miejscach i w różnej skali, a tło zmienia się z obrazu na obraz. Zaczniecie od MLP jako punktu odniesienia, a potem zobaczycie, co daje sieć splotowa: dzięki lokalnym filtrom ze współdzielonymi wagami ma kilka razy mniej parametrów i wyraźnie lepszą dokładność.

Zanim użyjecie gotowej warstwy `Conv2D`, policzycie splot ręcznie w numpy i sprawdzicie, że warstwa Keras robi dokładnie to samo. Przy okazji wyjaśni się, dlaczego to, co w uczeniu głębokim nazywa się splotem, jest w rzeczywistości korelacją wzajemną (cross-correlation).

Druga część zajęć dotyczy przeuczenia (overfitting). Większa sieć bez regularyzacji szybko zapamiętuje zbiór treningowy, więc porównacie cztery warianty: bez regularyzacji, z augmentacją danych, z dropoutem i L2 oraz ze wszystkim naraz. Na koniec zajrzycie do wnętrza sieci: obejrzycie filtry pierwszej warstwy i mapy cech (aktywacje) kolejnych warstw.

Treningi w zadaniach 4 i 5 trwają najdłużej. Uruchamiajcie je od razu po napisaniu kodu, a w czasie treningu odpowiadajcie na pytania z poprzednich zadań albo piszcie kod następnego. Zadanie 7 (mały ResNet) jest zadaniem rozszerzonym, opisanym na końcu instrukcji.

## Przed zajęciami

Przeczytajcie `lab02_wprowadzenie.md`, gdzie są objaśnione splot, dopełnianie, pooling, przeuczenie i metody regularyzacji. Przypomnijcie sobie z laboratorium 1 budowę modelu w Keras, `compile`, `fit` i krzywe uczenia. Otwórzcie `lab02_szablon.ipynb` w Google Colab i od razu włączcie GPU (Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU). Komórka z importami wypisuje wersje bibliotek i informację, czy GPU jest dostępne. Jeśli widzicie „brak, obliczenia na CPU”, zmieńcie typ środowiska, zanim zaczniecie trenować.

Na początku notebooka jest komórka konfiguracyjna z rozmiarem zbioru treningowego (`N_TRAIN`) i liczbami epok dla poszczególnych zadań. Wartości domyślne są przeznaczone dla Colaba z GPU, ale czas treningu w Colabie nie był mierzony. Jeśli treningi w zadaniu 5 będą trwały za długo, zmniejszcie `N_TRAIN` (na przykład do 20 000) albo liczbę epok i zapiszcie we wnioskach, co zmieniliście. Wszystkie dalsze komórki korzystają z tych stałych.

Gotowe w szablonie są: wczytanie CIFAR-10 i polskie nazwy klas (`KLASY`), funkcje rysujące (`pokaz_siatke`, `rysuj_krzywe`, `pokaz_obrazy`, `pokaz_filtry`, `pokaz_mapy`), funkcja `podsumuj`, która z historii treningu robi wiersz tabeli wyników, oraz obraz i filtry do zadania 2.

## Zadanie 1: Dane i baseline MLP

1. Wybierzcie po dwa przykłady z każdej klasy i pokażcie je funkcją `pokaz_siatke` z nazwami klas.
2. Znormalizujcie piksele do zakresu [0, 1] (typ `float32`). Wydzielcie zbiór walidacyjny: ostatnie `N_VAL` (5 000) obrazów z części treningowej. Zbiór treningowy to pierwsze `N_TRAIN` obrazów. Zbiór testowy też znormalizujcie, ale nie używajcie go aż do zadania 5. Wypiszcie kształty i liczność klas (`np.bincount`).
3. Zbudujcie MLP: `Flatten`, `Dense(256, relu)`, `Dense(128, relu)`, `Dense(10, softmax)`. Napiszcie funkcję `kompiluj(model)` (Adam z learning rate 1e-3, strata `sparse_categorical_crossentropy`, metryka accuracy) i funkcję `trenuj(model, epoki)`, która wywołuje `fit` z walidacją, `batch_size=BATCH` i `verbose=2` i zwraca historię oraz czas treningu w sekundach. Obu funkcji użyjecie we wszystkich kolejnych zadaniach.
4. Przed zbudowaniem modelu ustawcie ziarno (`keras.utils.set_random_seed(RANDOM_STATE)`). Wytrenujcie MLP przez `EPOKI_MLP` epok, wypiszcie `summary()` i dodajcie wiersz `podsumuj(...)` do listy `wyniki`. Narysujcie krzywe uczenia.

Skąd bierze się większość parametrów MLP? Jaki wynik dałoby losowe zgadywanie przy dziesięciu zrównoważonych klasach? Co MLP „wie” o tym, że sąsiednie piksele są ze sobą powiązane, i co by się stało z jego wynikiem, gdyby wszystkie obrazy przepuścić przez tę samą losową permutację pikseli? Dlaczego walidację wydzielamy z części treningowej, a nie używamy zbioru testowego?

## Zadanie 2: Splot ręcznie

Gotowa komórka wybiera pierwszą ciężarówkę ze zbioru treningowego, zamienia ją na skalę szarości (tablica `obraz` 32x32) i definiuje słownik `FILTRY` z trzema filtrami 3x3: Sobel Gy (reaguje na krawędzie poziome), Sobel Gx (krawędzie pionowe) i rozmycie (średnia z okna 3x3).

1. Napiszcie `splot2d(obraz, filtr)` bez dopełniania (`valid`) i z krokiem 1. Wynik ma rozmiar `(H - kh + 1, W - kw + 1)`, a każdy jego element to suma iloczynów okna obrazu i filtra: `wynik[i, j] = suma(obraz[i:i+kh, j:j+kw] * filtr)`. Dwie zagnieżdżone pętle w zupełności wystarczą.
2. Zastosujcie trzy filtry i pokażcie oryginał oraz wyniki funkcją `pokaz_obrazy`.
3. Utwórzcie `layers.Conv2D(3, 3, padding="valid", use_bias=False)` i zbudujcie ją dla wejścia o kształcie `(1, 32, 32, 1)`. Sprawdźcie kształt wag (`get_weights()[0]`), ułóżcie trzy filtry w tablicę o tym kształcie i ustawcie ją przez `set_weights`. Przepuśćcie obraz przez warstwę i porównajcie wynik każdego kanału z wynikiem numpy (maksymalna różnica bezwzględna).
4. Porównajcie wynik `splot2d` z matematycznym splotem `scipy.signal.convolve2d(obraz, filtr, mode="valid")`, a potem wynik `splot2d` z filtrem obróconym o 180 stopni (`filtr[::-1, ::-1]`) z tym samym splotem.

Gdzie na obrazie Sobel Gy daje duże wartości dodatnie i ujemne, a gdzie Sobel Gx? Dlaczego wynik ma rozmiar 30x30 i co trzeba by zrobić, żeby został 32x32? Czym różni się korelacja wzajemna od splotu i dlaczego w sieci neuronowej nie ma to praktycznego znaczenia? Dla którego z trzech filtrów oba działania dają ten sam wynik i dlaczego?

## Zadanie 3: Pierwsza sieć splotowa

1. Zbudujcie sieć: `Conv2D(32, 3, relu)`, `MaxPooling2D(2)`, `Conv2D(64, 3, relu)`, `MaxPooling2D(2)`, `Conv2D(128, 3, relu)`, `MaxPooling2D(2)`, `Flatten`, `Dense(64, relu)`, `Dense(10, softmax)`. Zostawcie domyślne `padding="valid"`. Warstwy splotowe nazwijcie `conv1`, `conv2`, `conv3`.
2. Policzcie ręcznie liczbę parametrów `conv1` i `conv2`: każdy filtr ma `k * k * kanały_wejścia` wag i jeden bias. Porównajcie z `count_params()` tych warstw.
3. Zanim cokolwiek wypiszecie, przewidźcie rozmiary map cech po każdej warstwie (splot 3x3 bez dopełniania zmniejsza wysokość i szerokość o 2, pooling 2x2 dzieli je przez 2 z zaokrągleniem w dół). Potem zbudujcie tabelę z nazwą warstwy, `layer.output.shape` i liczbą parametrów każdej warstwy i sprawdźcie swoje przewidywania.
4. Wytrenujcie sieć przez `EPOKI_CNN` epok, dodajcie wiersz do `wyniki` i porównajcie z MLP: liczba parametrów, dokładność walidacyjna, czas epoki. Narysujcie krzywe uczenia obu modeli na jednym wykresie.

Od czego zależy liczba parametrów warstwy `Conv2D`, a od czego nie zależy? Dlaczego CNN ma kilka razy mniej parametrów niż MLP, a działa lepiej? W której warstwie CNN jest najwięcej parametrów? Co dzieje się z rozmiarem przestrzennym i liczbą kanałów w kolejnych warstwach i dlaczego po drugim poolingu mapa ma 6, a nie 6,5 piksela?

## Zadanie 4: Przeuczenie

1. Napiszcie funkcję `zbuduj_cnn(augmentacja=None, dropout=0.0, l2=0.0, nazwa="cnn")`. Architektura bazowa ma trzy etapy. W każdym są dwie warstwy `Conv2D(f, 3, padding="same", relu)` i `MaxPooling2D(2)`, kolejno dla f = 32, 64 i 128 (nazwy `conv1a`, `conv1b`, `conv2a`, ..., `conv3b`). Potem `Flatten`, `Dense(256, relu)` (nazwa `dense`) i `Dense(10, softmax)`. Parametry `augmentacja`, `dropout` i `l2` wykorzystacie w zadaniu 5; już teraz napiszcie funkcję tak, by przy wartościach domyślnych budowała sieć bez regularyzacji. Wskazówki są w komentarzu w szablonie.
2. Wytrenujcie sieć bez regularyzacji przez `EPOKI_DUZA` epok i narysujcie krzywe uczenia (strata i dokładność).
3. Znajdźcie epokę, w której strata walidacyjna jest najmniejsza. Wypiszcie dokładność i stratę treningową oraz walidacyjną w tej epoce i w ostatniej.

W której epoce zaczyna się przeuczenie i po czym to widać na obu wykresach? Dlaczego strata walidacyjna rośnie wyraźnie, a dokładność walidacyjna spada powoli albo prawie stoi? Jak na moment przeuczenia wpłynęłoby zmniejszenie albo zwiększenie liczby obrazów treningowych?

## Zadanie 5: Regularyzacja

1. Napiszcie `zbuduj_augmentacje()`, która zwraca `keras.Sequential` z warstwami `RandomFlip("horizontal")`, `RandomTranslation(0.1, 0.1)` i `RandomRotation(0.05)`. Te warstwy są częścią modelu: losowo zmieniają obraz tylko podczas treningu, a przy predykcji przepuszczają go bez zmian. Pokażcie osiem wersji jednego obrazu treningowego, wywołując augmentację z `training=True`.
2. Porównajcie cztery warianty sieci z zadania 4:
   - A: bez regularyzacji, czyli model z zadania 4 (nie trenujcie go drugi raz),
   - B: augmentacja,
   - C: dropout 0,25 po każdym poolingu i 0,5 przed ostatnią warstwą oraz L2 o współczynniku 1e-4 na wagach warstw `Conv2D` i `Dense` (`kernel_regularizer`),
   - D: augmentacja, dropout i L2 razem.

   Warianty B, C i D trenujcie przez `EPOKI_DUZA` epok. Przed każdym ustawcie to samo ziarno i dla każdego modelu z augmentacją utwórzcie nowy obiekt `zbuduj_augmentacje()`.
3. Zbudujcie tabelę wyników walidacyjnych (wiersze z `podsumuj` i dodatkowa kolumna z różnicą dokładności treningowej i walidacyjnej po ostatniej epoce). Narysujcie dwa panele: `val_loss` i `val_accuracy` czterech wariantów w funkcji epoki.
4. Wybierzcie najlepszy wariant według dokładności walidacyjnej po ostatniej epoce. Dopiero teraz, i tylko dla tego modelu, policzcie dokładność na zbiorze testowym (`evaluate` na `x_te`).

Który wariant wygrał i o ile? Jak regularyzacja zmieniła różnicę między dokładnością treningową a walidacyjną i położenie minimum straty walidacyjnej? Dlaczego w wariantach z augmentacją albo dropoutem dokładność treningowa bywa niższa od walidacyjnej? Strata z L2 zawiera karę za wielkość wag: czy można wprost porównywać `val_loss` wariantów A i C? Czy krzywe wariantów z regularyzacją już się wypłaszczyły? Dlaczego zbiór testowy liczymy tylko raz i tylko dla wybranego modelu?

## Zadanie 6: Co widzi sieć

1. Weźcie najlepszy model z zadania 5 i pokażcie 32 filtry jego pierwszej warstwy splotowej (`conv1a`, wagi o kształcie `(3, 3, 3, 32)`) funkcją `pokaz_filtry`. Każdy filtr jest skalowany osobno do zakresu [0, 1] i wyświetlany jako mały obraz RGB.
2. Zbudujcie model pomocniczy, który dla obrazu wejściowego zwraca aktywacje warstw `conv1a`, `conv2b` i `conv3b`: `keras.Model(model.inputs, [model.get_layer(n).output for n in nazwy])`. Wybierzcie jeden obraz walidacyjny (na przykład pierwszy samochód), pokażcie go razem z predykcją modelu i narysujcie po 16 map cech z każdej z trzech warstw funkcją `pokaz_mapy`.
3. Przepuśćcie przez model pomocniczy 500 obrazów walidacyjnych i dla każdej z trzech warstw policzcie kanały, które ani razu się nie aktywowały (maksimum po wszystkich obrazach i pozycjach równe 0).

Jakie wzorce widać w filtrach pierwszej warstwy (kolory, kierunki krawędzi) i jak mają się do filtrów z zadania 2? Czym różnią się mapy cech kolejnych warstw pod względem rozdzielczości i czytelności dla człowieka? Co oznaczają kanały, które nigdy się nie aktywują, i czy to problem?

## Pytania kontrolne

1. Dlaczego warstwa splotowa ma tyle samo parametrów dla obrazu 32x32 i dla obrazu 1000x1000, a warstwa gęsta po `Flatten` nie?
2. Co to znaczy, że splot jest ekwiwariantny względem przesunięcia, a co daje sieci pooling?
3. Policzcie liczbę parametrów warstwy `Conv2D(64, 5)` działającej na wejściu o 32 kanałach.
4. Po czym na krzywych uczenia rozpoznacie przeuczenie, a po czym niedouczenie (underfitting)?
5. Dlaczego augmentacja danych jest w praktyce jedną z najskuteczniejszych metod regularyzacji dla obrazów? Podajcie przykład przekształcenia, które dla CIFAR-10 byłoby szkodliwe.
6. Czym różni się działanie warstwy `Dropout` podczas treningu i podczas predykcji?
7. Dlaczego wynik na zbiorze testowym nie powinien wpływać na wybór architektury ani hiperparametrów?

## Zadania rozszerzone

Zadanie 7 wykracza poza podstawowy zakres laboratorium. Wymaga funkcji `kompiluj`, `trenuj`, `zbuduj_augmentacje` i wyników zadania 5.

### Zadanie 7: Mały ResNet

Głębokie sieci zbudowane z samych warstw `Conv2D` trenuje się coraz trudniej, im są głębsze. ResNet rozwiązuje to połączeniami rezydualnymi (skip connections): wyjście bloku to suma jego wejścia i wyniku dwóch splotów, więc blok uczy się tylko poprawki do tego, co już dostał. Razem z normalizacją wsadową (`BatchNormalization`) pozwala to trenować sieci o dziesiątkach i setkach warstw.

1. Napiszcie `blok_rezydualny(x, filtry, krok=1, nazwa="blok")`: `Conv2D(filtry, 3, strides=krok, padding="same", use_bias=False)`, `BatchNormalization`, ReLU, `Conv2D(filtry, 3, padding="same", use_bias=False)`, `BatchNormalization`, dodanie skrótu warstwą `Add` i ReLU. Gdy krok jest większy od 1 albo liczba kanałów wejścia różni się od `filtry`, skrót przechodzi przez `Conv2D(filtry, 1, strides=krok, use_bias=False)` i `BatchNormalization`, bo inaczej kształty nie pozwolą na dodawanie.
2. Napiszcie `zbuduj_resnet(glowa="gap", augmentacja=None)`: wejście, opcjonalnie augmentacja, `Conv2D(32, 3, padding="same", use_bias=False)` z `BatchNormalization` i ReLU, trzy etapy po jednym bloku rezydualnym (32 kanały z krokiem 1, 64 i 128 z krokiem 2), na końcu `GlobalAveragePooling2D` (dla `glowa="gap"`) albo `Flatten` (dla `glowa="flatten"`) i `Dense(10, softmax)` o nazwie `wyjscie`.
3. Zbudujcie obie wersje bez treningu i zestawcie w tabeli: rozmiar wejścia warstwy `wyjscie`, jej liczbę parametrów i liczbę parametrów całego modelu. Dodajcie wiersz dla sieci z zadania 4, gdzie głowa to `Flatten`, `Dense(256)` i `Dense(10)`.
4. Wytrenujcie ResNet z `GlobalAveragePooling2D` i z augmentacją z zadania 5 przez `EPOKI_RESNET` epok. Porównajcie go z najlepszą CNN z zadania 5: liczba parametrów, czas epoki, dokładność walidacyjna po ostatniej epoce oraz dokładność walidacyjna CNN po tej samej liczbie epok, którą trenowaliście ResNet. Narysujcie krzywe obu modeli. Na koniec policzcie raz dokładność testową ResNetu.

Ile parametrów oszczędza `GlobalAveragePooling2D` w porównaniu z `Flatten` i jaką informację traci? Po co skrót 1x1 przy zmianie liczby kanałów i kroku? Jak ResNet wypada przy tej samej liczbie epok co najlepsza CNN, przy jakiej liczbie parametrów i jakim czasie epoki? Jaką rolę grają `BatchNormalization` i połączenia rezydualne i dlaczego przy trzech blokach ich zaleta może być jeszcze słabo widoczna?

## Co oddać

Notebook `lab02_szablon.ipynb` po uzupełnieniu, z zapisanymi wynikami komórek, wykresami i odpowiedziami na pytania w komórkach „Odpowiedzi”, w formie wskazanej przez prowadzącego. Przed oddaniem uruchomcie notebook od początku do końca (Środowisko wykonawcze, Uruchom wszystko). Jeśli zmieniliście wartości w komórce konfiguracyjnej, napiszcie to we wnioskach. Jeśli robicie zadanie rozszerzone, oddajecie ten sam notebook z uzupełnioną sekcją „Zadania rozszerzone”.

## Źródła

- F. Chollet, Deep Learning with Python, 2nd ed., Manning 2021: rozdział 5 (przeuczenie, regularyzacja), rozdział 8 (wprowadzenie do sieci splotowych, augmentacja), rozdział 9 (połączenia rezydualne, normalizacja wsadowa, interpretacja tego, czego uczą się sieci splotowe).
- I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press 2016 (www.deeplearningbook.org): rozdział 7 (regularyzacja), rozdział 9 (sieci splotowe, w tym splot i korelacja wzajemna).
- J. Krohn, G. Beyleveld, A. Bassens, Uczenie głębokie i sztuczna inteligencja, Helion 2021.
- Dokumentacja Keras: `https://keras.io/api/datasets/cifar10/`, `https://keras.io/api/layers/convolution_layers/convolution2d/`, `https://keras.io/api/layers/preprocessing_layers/image_augmentation/`.
- A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, raport techniczny, University of Toronto 2009 (zbiór CIFAR-10, `https://www.cs.toronto.edu/~kriz/cifar.html`).
- N. Srivastava i in., Dropout: A Simple Way to Prevent Neural Networks from Overfitting, Journal of Machine Learning Research 15, 2014.
- S. Ioffe, C. Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, ICML 2015.
- K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, CVPR 2016.
- M. Lin, Q. Chen, S. Yan, Network In Network, ICLR 2014 (global average pooling).
