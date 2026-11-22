# Laboratorium 4: Tuning i ewaluacja sieci neuronowych

Dane: CIFAR-10 (`keras.datasets.cifar10`, jak w laboratorium 2) oraz, w zadaniu 5, model MobileNetV2 z laboratorium 3 (`lab03_pets.keras`) i kilkaset zdjęć testowych Oxford-IIIT Pet
Środowisko: Google Colab z GPU (T4), Keras 3 na backendzie TensorFlow, KerasTuner (instalowany w pierwszej komórce), scikit-learn, pandas, matplotlib

## Cel

W poprzednich laboratoriach liczbę epok, learning rate i rozmiar sieci ustalaliście z góry. Tym razem zajmiecie się tym, jak takie decyzje podejmować w sposób uporządkowany i jak potem uczciwie ocenić wybrany model.

Zaczniecie od callbacków: `EarlyStopping` sam zatrzyma trening i przywróci najlepsze wagi, `ModelCheckpoint` zapisze najlepszy model do pliku, a `ReduceLROnPlateau` zmniejszy learning rate, gdy strata walidacyjna przestanie spadać. Potem porównacie trzy optymalizatory przy tej samej architekturze i tym samym budżecie epok i zobaczycie, co dzieje się przy za dużym learning rate. Trzecim krokiem jest automatyczne przeszukiwanie hiperparametrów KerasTunerem (RandomSearch), z wyborem najlepszej konfiguracji wyłącznie na zbiorze walidacyjnym.

Dopiero wybrany model oceniacie na zbiorze testowym, raz: macierz pomyłek, `classification_report`, różnica między dokładnością a macro F1 i najpewniejsze błędy. Na koniec metodą Grad-CAM sprawdzicie, na które fragmenty obrazu patrzy sieć, najpierw na własnej CNN z CIFAR-10, a potem na modelu MobileNetV2 wytrenowanym w laboratorium 3.

Najdłużej trwa przeszukiwanie w zadaniu 3 i ponowny trening najlepszej konfiguracji. Uruchamiajcie treningi od razu po napisaniu kodu, a w czasie treningu odpowiadajcie na pytania z poprzednich zadań albo piszcie kod następnego. Zadania 6 (Hyperband) i 7 (zmienność między seedami) są zadaniami rozszerzonymi, opisanymi na końcu instrukcji.

## Przed zajęciami

Przeczytajcie `lab04_wprowadzenie.md`. Znajdziecie tam opis callbacków, optymalizatorów, metod przeszukiwania hiperparametrów, metryk klasyfikacji i metody Grad-CAM.

Otwórzcie `lab04_szablon.ipynb` w Google Colab i włączcie GPU (Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU). Pierwsza komórka kodu instaluje `keras-tuner` i `tensorboard`. KerasTuner nie jest w Colabie preinstalowany, a przy backendzie TensorFlow importuje tensorboard w każdej próbie, więc bez niego wszystkie próby kończą się błędem. Komórka z importami wypisuje wersje bibliotek i informację, czy GPU jest dostępne.

Do zadania 5 (część B) potrzebny jest plik `lab03_pets.keras` z laboratorium 3 (około 24 MB). Jeśli go zapisaliście, wgrajcie go do bieżącego katalogu Colaba (panel Pliki po lewej stronie, Prześlij do magazynu sesji) albo zamontujcie Dysk Google i skopiujcie plik (przykład jest w komentarzu w komórce). Bez tego pliku zadanie 5 wykonujecie w części A, na własnej CNN z CIFAR-10, a część B jest pomijana automatycznie. Zdjęcia Oxford-IIIT Pet komórka pobiera sama z Hugging Face (plik testowy, około 400 MB, więc pobieranie chwilę trwa).

Na początku notebooka jest komórka konfiguracyjna z przełącznikiem `TRYB_SKROCONY`. W szablonie ma wartość `False`, czyli wartości dla Colaba z GPU: 20 000 obrazów treningowych (mniej niż 45 000 z laboratorium 2, bo tuner trenuje wiele modeli), do 40 epok w zadaniu 1, 15 epok w zadaniu 2, 8 prób po 9 epok w tunerze. Czasu w Colabie nie mierzono. Jeśli treningi trwają za długo, zmniejszcie `N_TRAIN`, `MAX_PROB` albo liczby epok i zapiszcie we wnioskach, co zmieniliście. `TRYB_SKROCONY = True` to wersja, w której wykonano rozwiązanie wzorcowe na CPU.

Gotowe w szablonie są: wczytanie i podział CIFAR-10 (`x_tr`, `x_val`, `x_te` w zakresie [0, 1]; walidacja to ostatnie `N_VAL` obrazów części treningowej), funkcja `zbuduj_cnn(filtry, dropout)` budująca małą CNN w stylu laboratorium 2, funkcje `podsumuj`, `rysuj_krzywe`, `rysuj_przebieg`, `pokaz_siatke`, `rysuj_macierz`, `tabela_prob`, callback `LicznikEpok` (liczy epoki zużyte przez tuner), a w zadaniu 5 funkcje `podziel_model` i `pokaz_gradcam` oraz wczytanie modelu i zdjęć z laboratorium 3.

Uwaga do Keras 3: argumenty `compile` podawajcie nazwane (`optimizer=..., loss=..., metrics=[...]`). Przy argumentach pozycyjnych Keras 3 zgłasza błąd, który nie wskazuje wprost przyczyny.

## Zadanie 1: Callbacki

1. Zbudujcie model `zbuduj_cnn(filtry=32, dropout=0.25)` i skompilujcie go z Adamem (learning rate 1e-3, strata `sparse_categorical_crossentropy`, metryka accuracy).
2. Przygotujcie trzy callbacki: `EarlyStopping(monitor="val_loss", patience=PATIENCE_ES, restore_best_weights=True)`, `ModelCheckpoint` zapisujący najlepszy model do pliku `lab04_cnn_najlepszy.keras` w bieżącym katalogu (`monitor="val_loss"`, `save_best_only=True`) i `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_LR, min_lr=1e-5)`. Napiszcie funkcję, która za każdym razem tworzy nowe obiekty callbacków, bo w zadaniu 3 użyjecie ich ponownie.
3. Trenujcie najwyżej `EPOKI_MAX` epok z walidacją (`validation_data=(x_val, y_val)`). Zbiór testowy zostaje nietknięty aż do zadania 4.
4. Wyznaczcie najlepszą epokę (minimum `val_loss`), epokę zatrzymania (`stopped_epoch` callbacku EarlyStopping) i epoki, od których zmienił się learning rate (klucz `learning_rate` w historii). Narysujcie przebieg funkcją `rysuj_przebieg`, która zaznacza te epoki na wykresie.
5. Sprawdźcie, że model w pamięci po treningu i model wczytany z pliku checkpointu (`keras.models.load_model`) dają tę samą stratę walidacyjną co najlepsza epoka.

W której epoce trening się zatrzymał i dlaczego właśnie wtedy? Jakie wagi ma model po treningu? Co działo się z krzywymi po każdej redukcji learning rate? Co by się stało, gdyby EarlyStopping monitorował `accuracy` zamiast `val_loss`? Dlaczego `patience` w ReduceLROnPlateau powinna być mniejsza niż w EarlyStopping?

## Zadanie 2: Optymalizatory

1. Porównajcie trzy optymalizatory przy tej samej architekturze (`zbuduj_cnn(32, 0.25)`), tych samych wagach początkowych (to samo ziarno przed budową modelu) i tym samym budżecie `EPOKI_OPT` epok, bez callbacków: SGD bez momentum (learning rate 0,01), SGD z momentum 0,9 (learning rate 0,01) i Adam (learning rate 0,001).
2. Zbudujcie tabelę wyników (`podsumuj`) i wspólny wykres `val_loss` i `val_accuracy` trzech wariantów (`rysuj_krzywe`).
3. Wytrenujcie jeszcze raz model z Adamem, ale z learning rate 0,03, wypiszcie stratę i dokładność walidacyjną w każdej epoce i porównajcie krzywe z Adamem 0,001.

Który optymalizator najszybciej obniżył stratę walidacyjną i o ile różnią się wyniki po `EPOKI_OPT` epokach? Co daje momentum w porównaniu ze zwykłym SGD? Czy wynik tego porównania przesądza, że Adam jest lepszy od SGD z momentum? Co dzieje się przy za dużym learning rate i po czym to poznać w historii treningu?

## Zadanie 3: KerasTuner

1. Napiszcie `zbuduj_model_hp(hp)`, która przeszukuje trzy hiperparametry: liczbę filtrów (`hp.Choice("filtry", [16, 32, 64])`), dropout (`hp.Choice("dropout", [0.0, 0.25, 0.5])`) i learning rate Adama (`hp.Choice("learning_rate", [3e-4, 1e-3, 3e-3])`), i zwraca skompilowany model `zbuduj_cnn`. Przestrzeń ma 3 x 3 x 3 = 27 konfiguracji.
2. Utwórzcie `kt.RandomSearch` z `objective="val_accuracy"`, `max_trials=MAX_PROB`, `seed=RANDOM_STATE`, `directory=KATALOG_TUNERA`, `project_name="random_search"` i `overwrite=True`. Uruchomcie `search` na zbiorze treningowym z walidacją, po `EPOKI_TUNER` epok na próbę. Przekażcie `callbacks=[LicznikEpok()]` (wcześniej ustawcie `LicznikEpok.epoki = 0`), żeby policzyć zużyte epoki, i zmierzcie czas. KerasTuner kopiuje callbacki dla każdej próby, dlatego licznik jest atrybutem klasy, a nie obiektu.
3. Pokażcie tabelę prób (`tabela_prob`). Wynik próby to najlepsza `val_accuracy` spośród jej epok.
4. Weźcie najlepszą konfigurację (`get_best_hyperparameters`), wytrenujcie ją od nowa z callbackami z zadania 1 (bez checkpointu) przez najwyżej `EPOKI_MAX` epok i porównajcie na walidacji z modelem z zadania 1. Wybierzcie lepszy model według `val_accuracy` (zmienna `model_wybrany`). Testu nadal nie ruszacie.

Ile konfiguracji ze wszystkich możliwych sprawdził tuner i ile epok to kosztowało? Które hiperparametry najbardziej różnicują wyniki? Dlaczego najlepszy model z tunera trenujemy jeszcze raz, zamiast od razu go użyć? Na czym tuner opierał wybór i dlaczego nie może to być zbiór testowy? Czy różnica między dwiema najlepszymi próbami jest wiarygodna (wrócicie do tego w zadaniach rozszerzonych)?

## Zadanie 4: Ewaluacja na zbiorze testowym

1. Dla wybranego modelu policzcie raz predykcje na zbiorze testowym (`predict`). Wypiszcie dokładność i macro F1 (`f1_score(..., average="macro")`).
2. Narysujcie macierz pomyłek (`confusion_matrix`, `rysuj_macierz`) i wypiszcie pięć najczęściej mylonych par klas.
3. Wypiszcie `classification_report` z nazwami klas.
4. Pokażcie dziesięć najpewniejszych błędów: obrazy, na których model się pomylił, posortowane według prawdopodobieństwa przypisanego błędnej klasie.
5. Zbudujcie z wyników testu (bez ponownego `predict`) zbiór niezbalansowany: wszystkie obrazy klasy „samochód” i po 50 obrazów każdej innej klasy. Porównajcie dokładność i macro F1 waszego modelu oraz klasyfikatora, który zawsze odpowiada „samochód”.

Które klasy model myli najczęściej i czy te pomyłki są zrozumiałe? Dlaczego na pełnym zbiorze testowym dokładność i macro F1 są prawie równe, a na zbiorze niezbalansowanym nie? Co mówi precision, a co recall dla klasy „kot”? Co widać na najpewniejszych błędach (etykieta, tło, podobieństwo klas)? Dlaczego test liczymy tylko raz i dopiero teraz?

## Zadanie 5: Grad-CAM

Grad-CAM (Selvaraju i in., 2017) pokazuje, które miejsca obrazu najbardziej podniosły wynik wybranej klasy. Bierzemy mapy cech $A^k$ ostatniej warstwy splotowej (kanały $k$) i wynik klasy przed softmaxem (logit $y^c$). Waga kanału to średni gradient $\alpha_k = \frac{1}{Z}\sum_{i,j} \partial y^c / \partial A^k_{ij}$, a mapa to $\mathrm{ReLU}(\sum_k \alpha_k A^k)$, przeskalowana do rozmiaru obrazu.

Gradient liczymy przez `tf.GradientTape`, więc to jest kod dla backendu TensorFlow. Gotowa funkcja `podziel_model(model, nazwa_warstwy)` zwraca funkcję `przejscie(x)`, która przechodzi kolejno przez warstwy modelu do wskazanej warstwy (mapy cech), potem przez resztę głowy, a z ostatniej warstwy `Dense` liczy logity zamiast prawdopodobieństw. Jest to potrzebne, bo w modelu z laboratorium 3 cała baza MobileNetV2 jest jedną warstwą, więc jej wewnętrznej warstwy `out_relu` nie da się podpiąć wprost jako wyjścia modelu.

1. Napiszcie `grad_cam(przejscie, obraz, klasa=None)`: w `GradientTape` wywołajcie `przejscie` na obrazie z dodaną osią wsadu, weźcie logit klasy (domyślnie przewidzianej), policzcie gradient po mapach cech, uśrednijcie go po wysokości i szerokości, zważcie mapy, zsumujcie po kanałach, przepuśćcie przez ReLU i podzielcie przez maksimum.
2. Część A, własna CNN z CIFAR-10 (wykonujecie zawsze): `podziel_model(model_wybrany, "conv_ost")`, mapy 8x8. Sprawdźcie, że softmax z logitów daje te same prawdopodobieństwa co `predict`. Pokażcie Grad-CAM dla czterech losowych trafień i czterech najpewniejszych błędów z zadania 4.
3. Część B, MobileNetV2 z laboratorium 3 (gdy jest plik `lab03_pets.keras`): gotowa komórka wczytuje model i `N_PETS` zdjęć testowych Oxford-IIIT Pet przeskalowanych do 160x160. Model zawiera warstwę `Rescaling`, więc przyjmuje piksele 0..255, tak jak w laboratorium 3. Baza to warstwa `mobilenetv2_1.00_160`, której wyjście to mapy `out_relu` 5x5x1280, więc `podziel_model(model_pets, "mobilenetv2_1.00_160")`. Sprawdźcie zgodność predykcji, pokażcie Grad-CAM dla czterech trafień, a dla czterech najpewniejszych błędów mapy dla klasy przewidzianej i dla prawdziwej.

Na co patrzy własna CNN przy trafieniach, a na co przy błędach? Dlaczego mapy są tak zgrubne i co ogranicza ich rozdzielczość w obu modelach? Czym różnią się mapy dla klasy przewidzianej i prawdziwej przy błędach modelu z laboratorium 3? Dlaczego używamy logitu, a nie prawdopodobieństwa po softmaxie? Czego Grad-CAM nie mówi o działaniu sieci?

## Pytania kontrolne

1. Czym różnią się parametry modelu od hiperparametrów? Podajcie po trzy przykłady.
2. Do czego służy zbiór treningowy, walidacyjny i testowy i co się stanie, jeśli hiperparametry dobierzecie na zbiorze testowym?
3. Jak działa `EarlyStopping` z `restore_best_weights=True` i dlaczego jest formą regularyzacji?
4. Dlaczego przeszukiwanie losowe (random search) bywa skuteczniejsze od przeszukiwania siatki (grid search) przy tym samym budżecie prób?
5. Kiedy dokładność jest mylącą miarą i co zamiast niej warto podać?
6. Co mówi macierz pomyłek, czego nie mówi pojedyncza liczba?
7. Co pokazuje mapa Grad-CAM i dlaczego nie jest to pełne wyjaśnienie decyzji sieci?

## Zadania rozszerzone

Zadania 6 i 7 wykraczają poza podstawowy zakres laboratorium. Wymagają funkcji `zbuduj_model_hp`, tabeli `tabela_rs` i najlepszej konfiguracji `najlepsze_hp` z zadania 3. Sekcja `# Zadania rozszerzone` jest na końcu notebooka.

### Zadanie 6: Hyperband a RandomSearch

Hyperband (Li i in., 2018) zaczyna wiele konfiguracji z małą liczbą epok, odrzuca słabsze i kontynuuje tylko najlepsze. Przy tym samym budżecie może więc sprawdzić więcej konfiguracji niż RandomSearch, ale ocenia je po krótszym treningu.

1. Utwórzcie `kt.Hyperband` z tą samą funkcją `zbuduj_model_hp`, `objective="val_accuracy"`, `max_epochs=EPOKI_TUNER`, `factor=3`, `hyperband_iterations=1`, `seed=RANDOM_STATE`, w tym samym katalogu z `project_name="hyperband"` i `overwrite=True`. Uruchomcie `search` z `LicznikEpok` (liczbę epok wyznacza tu Hyperband, więc nie podajecie `epochs`).
2. Pokażcie tabelę prób. Kolumny `tuner/epochs` i `tuner/round` mówią, do której epoki trenowano próbę i w której rundzie.
3. Zbudujcie tabelę porównawczą obu tunerów: liczba prób, liczba różnych konfiguracji, łączna liczba epok, czas, najlepsza `val_accuracy` i najlepsza konfiguracja.

Czy budżety obu przeszukiwań są porównywalne w epokach? Ile konfiguracji sprawdził każdy tuner i ile z nich Hyperband trenował przez pełne `max_epochs`? Czy oba wskazały tę samą konfigurację? Na czym polega ryzyko odrzucania konfiguracji po jednej lub dwóch epokach?

### Zadanie 7: Zmienność wyniku między seedami

1. Wytrenujcie najlepszą konfigurację z RandomSearch (`najlepsze_hp`) trzy razy z ziarnami 42, 43 i 44, za każdym razem przez `EPOKI_TUNER` epok, bez callbacków, czyli tak samo jak w próbie tunera.
2. Dla każdego ziarna zapiszcie najlepszą `val_accuracy` spośród epok (tak liczy wynik KerasTuner) i `val_accuracy` po ostatniej epoce. Policzcie średnią i odchylenie standardowe z poprawką na próbę (`ddof=1`). Narysujcie krzywe `val_accuracy` trzech przebiegów.
3. Porównajcie odchylenie z różnicami między próbami w tabeli RandomSearch: między pierwszą a drugą próbą i między pierwszą a medianą.

Jak duża jest zmienność wyniku przy tej samej konfiguracji? Czy różnice między najlepszymi konfiguracjami z tunera są od niej wyraźnie większe? Co z tego wynika dla wyboru zwycięzcy na podstawie jednej próby? Jak zmienilibyście procedurę tuningu, mając więcej czasu obliczeniowego (na przykład `executions_per_trial`)?

## Co oddać

Notebook `lab04_szablon.ipynb` po uzupełnieniu, z zapisanymi wynikami komórek, wykresami i odpowiedziami na pytania w komórkach „Odpowiedzi”, w formie wskazanej przez prowadzącego. Przed oddaniem uruchomcie notebook od początku do końca (Środowisko wykonawcze, Uruchom wszystko). Napiszcie we wnioskach, czy część B zadania 5 była wykonana na modelu z laboratorium 3, i czy zmienialiście wartości w komórce konfiguracyjnej. Ostatnia komórka usuwa katalog tunera (`lab04_tuner`), plik checkpointu zostaje. Jeśli robicie zadania rozszerzone, oddajecie ten sam notebook z uzupełnioną sekcją „Zadania rozszerzone”.

## Źródła

- F. Chollet, Deep Learning with Python, 2nd ed., Manning 2021: rozdział 5 (ewaluacja modeli, zbiory walidacyjne), rozdział 7 (callbacki), rozdział 9 (wizualizacja map aktywacji klas, Grad-CAM), rozdział 13 (optymalizacja hiperparametrów z KerasTunerem).
- I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press 2016 (www.deeplearningbook.org): rozdział 7 (w tym early stopping), rozdział 8 (optymalizacja, momentum, metody adaptacyjne), rozdział 11 (metodologia praktyczna, dobór hiperparametrów).
- Dokumentacja Keras: `https://keras.io/api/callbacks/`, `https://keras.io/api/optimizers/`, `https://keras.io/keras_tuner/`.
- Dokumentacja scikit-learn: `https://scikit-learn.org/stable/modules/model_evaluation.html`.
- R. R. Selvaraju i in., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, ICCV 2017.
- D. P. Kingma, J. Ba, Adam: A Method for Stochastic Optimization, ICLR 2015.
- J. Bergstra, Y. Bengio, Random Search for Hyper-Parameter Optimization, Journal of Machine Learning Research 13, 2012.
- L. Li i in., Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization, Journal of Machine Learning Research 18, 2018.
- A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, raport techniczny, University of Toronto 2009 (zbiór CIFAR-10).
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, Cats and Dogs, CVPR 2012 (zbiór Oxford-IIIT Pet).
- M. Sandler i in., MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.
