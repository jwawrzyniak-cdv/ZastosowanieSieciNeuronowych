# Laboratorium 3: Transfer learning w klasyfikacji obrazów

Dane: Oxford-IIIT Pet (37 ras psów i kotów, 3680 zdjęć treningowych i 3669 testowych, licencja CC BY-SA 4.0), pliki parquet z Hugging Face (`timm/oxford-iiit-pet`); w zadaniach rozszerzonych także własne zdjęcia
Środowisko: Google Colab z GPU (T4), Keras 3 na backendzie TensorFlow, notebook `lab03_szablon.ipynb`

## Cel

Na laboratorium 2 sieć konwolucyjna uczyła się od zera na 50 tys. małych zdjęć CIFAR-10. Tym razem zdjęć jest około 3 tys., klas 37, a różnice między klasami bywają subtelne: dwie rasy terierów albo dwa długowłose koty. Sprawdzicie najpierw, ile da się osiągnąć małą siecią uczoną od zera, a potem użyjecie sieci MobileNetV2 wytrenowanej na ImageNet. Jej warstwy konwolucyjne nauczyły się już rozpoznawać krawędzie, tekstury, części zwierząt i przedmiotów. Wystarczy dołożyć nową głowę klasyfikacyjną i ją wytrenować (ekstraktor cech, feature extraction), a potem ostrożnie dostroić górne warstwy bazy (fine-tuning).

Po drodze zobaczycie kilka szczegółów, które w praktyce decydują o wyniku: przetwarzanie wejścia zgodne z tym, na którym uczono bazę, tryb wnioskowania warstw BatchNormalization, niski learning rate przy fine-tuningu i wybór modelu na zbiorze walidacyjnym z jednorazowym pomiarem na teście. Na koniec przeanalizujecie, które rasy model myli, i zapiszecie model do pliku. Ten plik będzie potrzebny w laboratorium 4, gdzie metodą Grad-CAM sprawdzicie, na które fragmenty zdjęcia patrzy sieć.

Zadania 6 i 7 na końcu instrukcji to zadania rozszerzone: porównanie kilku baz jako ekstraktorów cech i transfer learning na własnym małym zbiorze.

## Przed zajęciami

Przeczytajcie `lab03_wprowadzenie.md`. Opisuje, czym jest transfer learning, jak działa MobileNetV2, czym różni się ekstraktor cech od fine-tuningu i dlaczego warstwy BatchNormalization wymagają przy nim osobnej uwagi.

Otwórzcie `lab03_szablon.ipynb` w Google Colab i włączcie GPU: Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU. Bez GPU trening będzie kilka razy wolniejszy. Uruchomcie komórkę z importami i sprawdźcie, czy w wypisanej informacji jest GPU.

Komórka wczytująca dane w zadaniu 1 jest gotowa. Pobiera dwa pliki parquet (razem około 790 MB), dekoduje zdjęcia JPEG po kolei, skaluje je do 160x160 pikseli i zapisuje jako tablice `uint8`. Z części treningowej wydziela 20% zdjęć na walidację (stratyfikacja po rasach), część testowa zostaje osobno. Lista 37 nazw ras jest w kodzie w kolejności etykiet. Gotowe są też funkcje `pokaz_obrazy` i `rysuj_krzywe`, callback `CzasEpoki` mierzący czas epok oraz funkcja `nowa_augmentacja` z warstwami augmentacji (odbicie w poziomie, obrót, przybliżenie). Liczby epok są w komórce konfiguracyjnej na początku notebooka.

Jeśli chcecie zrobić zadanie rozszerzone 7, przygotujcie przed zajęciami własne zdjęcia (po około 30 na każdą z 3 kategorii).

## Zadanie 1: Dane

1. Uruchomcie komórkę wczytującą dane i sprawdźcie rozmiary tablic `X_train`, `X_val`, `X_test`.
2. Policzcie zdjęcia w każdej klasie w zbiorze treningowym, walidacyjnym i testowym (`np.bincount`). Narysujcie wykres słupkowy dla zbioru treningowego z nazwami ras na osi X.
3. Pokażcie siatkę 12 losowych zdjęć treningowych z nazwami ras.

Ile zdjęć przypada na jedną rasę i czy zbiór jest zbalansowany? Dlaczego trening od zera na około 3 tys. zdjęć i 37 klasach jest trudny, skoro w laboratorium 2 sieć od zera radziła sobie z CIFAR-10? Które rasy łatwo pomylić? Co tracimy, skalując każde zdjęcie do kwadratu bez zachowania proporcji?

## Zadanie 2: Baseline, mała CNN od zera

1. Zbudujcie w API funkcyjnym małą sieć: wejście 160x160x3, `nowa_augmentacja()`, `Rescaling(1/255)`, 3 do 4 bloków `Conv2D` (ReLU, `padding="same"`) i `MaxPooling2D`, `GlobalAveragePooling2D`, `Dropout`, `Dense(37, activation="softmax")`.
2. Skompilujcie ją z `Adam(1e-3)`, funkcją straty `sparse_categorical_crossentropy` i metryką `accuracy`. Wypiszcie liczbę parametrów.
3. Wytrenujcie ją przez `EPOKI_BASELINE` epok (15) z `validation_data=(X_val, y_val)` i callbackiem `CzasEpoki`. Wypiszcie najlepszą dokładność walidacyjną i średni czas epoki, narysujcie krzywe uczenia.

Nie zakładajcie z góry, jaki wynik wyjdzie, zmierzcie go. Ile razy lepiej od zgadywania (1/37) działa sieć? Czy jest przeuczona, czy niedouczona? Co mogłoby poprawić wynik sieci uczonej od zera i ile by to kosztowało?

## Zadanie 3: MobileNetV2 jako ekstraktor cech

1. Wczytajcie bazę: `keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(160, 160, 3))`. Przy pierwszym wywołaniu pobierają się wagi (około 9 MB). Zamroźcie ją: `baza.trainable = False`. Wypiszcie liczbę warstw i parametrów bazy oraz kształt jej wyjścia.
2. Zbudujcie model: wejście, `nowa_augmentacja()`, `Rescaling(1/127.5, offset=-1)`, `baza(x, training=False)`, `GlobalAveragePooling2D()`, `Dropout(0.2)`, `Dense(37, activation="softmax")`. Warstwa `Rescaling` przekształca piksele 0..255 do zakresu [-1, 1], czyli robi to samo co `keras.applications.mobilenet_v2.preprocess_input`, a jako część modelu zapisze się razem z nim.
3. Skompilujcie z `Adam(1e-3)`. Wypiszcie liczbę wszystkich parametrów i parametrów trenowanych.
4. Wytrenujcie głowę przez `EPOKI_GLOWA` epok (10) z gotowym callbackiem `checkpoint` (zapisuje wagi z najlepszą dokładnością walidacyjną do pliku `lab03_najlepsze.weights.h5`) i `CzasEpoki`. Narysujcie krzywe i zestawcie wynik z baseline w tabeli.

Zwróćcie uwagę na `training=False` przy wywołaniu bazy. MobileNetV2 zawiera warstwy BatchNormalization. W trybie treningowym normalizują one aktywacje statystykami bieżącego batcha i nadpisują średnie kroczące, zapamiętane podczas treningu na ImageNet. Przewodnik Keras zaleca wywołanie z `training=False`, żeby utrzymać je w trybie wnioskowania także po odmrożeniu części warstw w zadaniu 4. Bez tego pierwsze kroki fine-tuningu potrafią zepsuć cechy, które baza już ma.

Jest tu pułapka. W Keras 3.13.2, na którym przygotowano rozwiązanie, `fit` przekazuje `training=True` do zagnieżdżonej bazy i nadpisuje argument `training=False` (sprawdziliśmy to na małym przykładzie: średnie kroczące odmrożonej warstwy BatchNormalization zmieniały się). Dopóki baza jest zamrożona, nie ma to znaczenia, bo warstwa BatchNormalization z `trainable=False` zawsze działa w trybie wnioskowania. Dlatego w zadaniu 4 warstwy BatchNormalization zamrozicie jawnie.

Ile parametrów trenujecie teraz, a ile w CNN od zera? Skąd tak duża różnica w wyniku? Co by się stało z wynikiem, gdyby wejście przeskalować `Rescaling(1/255)` zamiast do [-1, 1]? Dlaczego dokładność walidacyjna bywa tu wyższa niż treningowa?

## Zadanie 4: Fine-tuning i pomiar na teście

1. Odmroźcie bazę (`baza.trainable = True`), a potem z powrotem zamroźcie dolne warstwy i wszystkie warstwy BatchNormalization: dla każdej warstwy bazy `warstwa.trainable = i >= FINE_TUNE_OD and not isinstance(warstwa, layers.BatchNormalization)`, gdzie `i` to jej indeks. Przy `FINE_TUNE_OD = 110` górna część bazy to 44 ostatnie warstwy ze 154, z czego 15 to BatchNormalization, więc odmrażacie 29 warstw.
2. Skompilujcie model ponownie z niskim learning rate, `Adam(1e-5)`. Zmiana `trainable` zaczyna działać dopiero po ponownej kompilacji. Wypiszcie liczbę parametrów trenowanych. Zapamiętajcie `moving_mean` ostatniej warstwy BatchNormalization i po treningu sprawdźcie, że się nie zmieniła.
3. Trenujcie dalej przez `EPOKI_FT` epok (10): `epochs=EPOKI_GLOWA + EPOKI_FT`, `initial_epoch=EPOKI_GLOWA`, z tym samym obiektem `checkpoint`. Ten sam obiekt pamięta najlepszy wynik z zadania 3, więc po obu etapach w pliku są wagi z najlepszej epoki całego treningu.
4. Połączcie historie obu etapów w jeden słownik i narysujcie krzywe z pionową linią w miejscu początku fine-tuningu (`rysuj_krzywe(..., granica=EPOKI_GLOWA)`).
5. Wczytajcie najlepsze wagi (`model.load_weights(PLIK_WAG)`), policzcie raz predykcje dla `X_test` i dokładność testową. Zestawcie w tabeli najlepszą dokładność walidacyjną trzech modeli: CNN od zera, ekstraktor cech, fine-tuning.

Dlaczego fine-tuning wymaga niskiego learning rate? Dlaczego nie odmrażamy bazy od razu, razem z losowo zainicjowaną głową? O ile fine-tuning poprawił wynik i czy przy tej liczbie epok było warto? Dlaczego zbiór testowy liczymy raz, dla jednego wybranego modelu?

## Zadanie 5: Analiza błędów i zapis modelu

1. Policzcie macierz pomyłek dla zbioru testowego (`sklearn.metrics.confusion_matrix`, predykcje z zadania 4) i narysujcie ją.
2. Wypiszcie 5 par ras mylonych najczęściej. Dla każdej pary ras `i < j` zsumujcie pomyłki w obie strony (`cm[i, j] + cm[j, i]`) i posortujcie malejąco.
3. Pokażcie 12 błędnie sklasyfikowanych zdjęć testowych z prawdziwą i przewidzianą rasą.
4. Zapiszcie model: `model.save("lab03_pets.keras")`. Wczytajcie go `keras.models.load_model("lab03_pets.keras")` i sprawdźcie, że daje te same predykcje.
5. Zachowajcie plik `lab03_pets.keras`, bo będzie potrzebny w laboratorium 4. Pliki w Colabie znikają po zakończeniu sesji. Pobierzcie go na swój komputer (panel Pliki, menu przy pliku, Pobierz, albo `from google.colab import files; files.download("lab03_pets.keras")`) lub skopiujcie na Google Drive po zamontowaniu dysku (`from google.colab import drive; drive.mount("/content/drive")`).

Czy najczęściej mylone pary mają sens, gdy spojrzycie na zdjęcia tych ras? Czy model myli psa z kotem? Czego nie widać w samej dokładności, a widać w macierzy pomyłek? Co zrobilibyście dalej, żeby poprawić model?

## Pytania kontrolne

1. Czym różni się ekstraktor cech od fine-tuningu? Które wagi są trenowane w każdym z tych podejść?
2. Dlaczego przy transfer learningu przetwarzanie wejścia musi odpowiadać temu, którego użyto przy treningu bazy?
3. Co robi argument `training=False` przy wywołaniu bazy, dlaczego ma znaczenie dla warstw BatchNormalization i dlaczego przy fine-tuningu bezpieczniej zamrozić te warstwy jawnie?
4. Dlaczego fine-tuning zaczyna się dopiero po wytrenowaniu głowy i z learning rate mniejszym o rząd lub dwa?
5. Dlaczego niższe warstwy sieci konwolucyjnej przenoszą się między zadaniami lepiej niż wyższe?
6. Po co jest osobny zbiór walidacyjny, skoro mamy zbiór testowy?

## Zadania rozszerzone

Zadania 6 i 7 wykraczają poza podstawowy zakres laboratorium.

### Zadanie 6: Porównanie baz jako ekstraktorów cech

1. Napiszcie funkcję `zbuduj_ekstraktor(baza, przetworzenie, liczba_klas)`, która buduje i kompiluje model z tą samą głową co w zadaniu 3 (augmentacja, przetworzenie wejścia, `baza(x, training=False)`, `GlobalAveragePooling2D`, `Dropout(0.2)`, `Dense`, `Adam(1e-3)`).
2. Wytrenujcie trzy modele z zamrożonymi bazami `MobileNetV2`, `EfficientNetB0` i `ResNet50` (wszystkie z `include_top=False`, `weights="imagenet"`, `input_shape=(160, 160, 3)`). Budżet ma być identyczny: `EPOKI_POROWNANIE` epok (5), ten sam batch, te same dane, ten sam seed przed każdym modelem.
3. Każda baza oczekuje innego wejścia. MobileNetV2: zakres [-1, 1] (`Rescaling(1/127.5, offset=-1)`). EfficientNetB0 w Keras ma skalowanie i normalizację wbudowane w model, więc dostaje piksele 0..255 bez żadnego przetwarzania. ResNet50: `keras.applications.resnet50.preprocess_input`, które zamienia kolejność kanałów na BGR i odejmuje średnie z ImageNet.
4. Zestawcie w tabeli: najlepszą dokładność walidacyjną, liczbę parametrów bazy, liczbę parametrów trenowanych i średni czas epoki. Pierwsza epoka zawiera kompilację grafu, więc podajcie też średnią bez niej.

Która baza wypadła najlepiej przy tym samym budżecie i czy wynik idzie w parze z liczbą parametrów? Którą wybralibyście do aplikacji na telefonie, a którą do serwera z GPU? Co by się stało, gdyby EfficientNetB0 dostała wejście w zakresie [-1, 1]?

### Zadanie 7: Własny mały zbiór

1. Zróbcie telefonem lub zbierzcie po około 30 zdjęć dla 3 kategorii, na przykład trzech własnych przedmiotów. Ułóżcie je w katalogach `moje_zdjecia/kategoria_a/`, `moje_zdjecia/kategoria_b/`, `moje_zdjecia/kategoria_c/`, spakujcie do `moje_zdjecia.zip`, wgrajcie do Colaba przez panel Pliki i rozpakujcie: `!unzip -q moje_zdjecia.zip`.
2. Wczytajcie zdjęcia: `keras.utils.image_dataset_from_directory("moje_zdjecia", validation_split=0.2, subset="both", seed=RANDOM_STATE, image_size=(160, 160), batch_size=16)`. Nazwy klas biorą się z nazw katalogów (`class_names`).
3. Zbudujcie ekstraktor cech MobileNetV2 dla 3 klas funkcją z zadania 6 i wytrenujcie go przez `EPOKI_MINI` epok (15).
4. Oceńcie model na zbiorze walidacyjnym i pokażcie kilka zdjęć walidacyjnych z predykcjami. Spróbujcie też kilku nowych zdjęć zrobionych w innym miejscu.

Kto nie ma własnych zdjęć, może użyć gotowej funkcji `przygotuj_zbior_zastepczy`, która zapisuje po 30 zdjęć z 3 ras Oxford Pets w tej samej strukturze katalogów.

Jak bardzo możecie ufać dokładności policzonej na kilkunastu zdjęciach walidacyjnych? Dlaczego transfer learning działa przy 30 zdjęciach na klasę, a CNN od zera z zadania 2 nie radziła sobie przy około 80? Jakie cechy waszych zdjęć (to samo tło, oświetlenie, ten sam stół) mogą sprawić, że model uczy się czegoś innego, niż chcecie?

## Co oddać

Notebook z wynikami wszystkich komórek i odpowiedziami na pytania do przemyślenia przy każdym zadaniu oraz na pytania kontrolne, w formie wskazanej przez prowadzącego. Plik `lab03_pets.keras` zachowajcie u siebie do laboratorium 4. Jeśli robicie zadania rozszerzone, dołączcie zadania 6 i 7 (bez własnych zdjęć, wystarczą wyniki w notebooku).

## Źródła

- F. Chollet, Deep Learning with Python, 2nd ed., Manning 2021, rozdział 8 (podrozdział o wykorzystaniu sieci wytrenowanej wcześniej: ekstrakcja cech i fine-tuning) i rozdział 9 (interpretacja sieci konwolucyjnych, Grad-CAM).
- I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press 2016, rozdział 9 (sieci konwolucyjne) i podrozdział 15.2 (transfer learning i adaptacja dziedziny), www.deeplearningbook.org.
- Dokumentacja Keras: przewodnik „Transfer learning & fine-tuning” (https://keras.io/guides/transfer_learning/), Keras Applications (https://keras.io/api/applications/), wczytywanie obrazów z katalogów (https://keras.io/api/data_loading/image/).
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, Cats and Dogs, CVPR 2012 (zbiór Oxford-IIIT Pet, https://www.robots.ox.ac.uk/~vgg/data/pets/).
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, MobileNetV2: Inverted Residuals and Linear Bottlenecks, CVPR 2018.
- M. Tan, Q. V. Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, ICML 2019.
- K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, CVPR 2016.
- J. Yosinski, J. Clune, Y. Bengio, H. Lipson, How transferable are features in deep neural networks?, NeurIPS 2014.
