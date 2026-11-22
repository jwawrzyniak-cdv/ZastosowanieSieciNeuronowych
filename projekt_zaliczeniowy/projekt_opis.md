# Projekt zaliczeniowy: sieć neuronowa w wybranym zastosowaniu

## Założenia

Sylabus modułu "Zaawansowana sztuczna inteligencja" mówi, że podstawą zaliczenia kursu "Zastosowanie sieci neuronowych" jest zadanie realizowane w trakcie zajęć lub na końcu, dotyczące praktycznych zastosowań omawianych zagadnień. Ocena z kursu stanowi 50% oceny z modułu. Ten projekt jest zadaniem "na końcu". Zadania z laboratoriów 1 do 6 są zadaniami "w trakcie zajęć". Sposób połączenia obu ocen podaje prowadzący na pierwszych zajęciach.

Zadanie polega na rozwiązaniu jednego problemu klasyfikacji (albo regresji) obrazów lub tekstu za pomocą sieci neuronowej i rzetelnym sprawdzeniu, jak dobre jest to rozwiązanie. Liczy się nie tylko wynik, ale cała droga: punkt odniesienia, kolejne modele, eksperymenty z hiperparametrami, uczciwy test i analiza błędów. Na laboratoriach ćwiczyliście każdy z tych kroków osobno. W projekcie składacie je w całość na danych, których nie widzieliście na zajęciach.

Projekt wykonujecie indywidualnie. Na obronie trzeba umieć wyjaśnić cały kod i każdą decyzję, także te, przy których korzystaliście z gotowych przykładów albo pomocy narzędzi.

Środowisko pracy to Google Colab z GPU. Wynikiem jest notebook, który przechodzi od początku do końca po wybraniu "Uruchom wszystko", bez ręcznego wgrywania plików. Dane pobiera kod z publicznego adresu. Jeżeli pełny trening trwa dłużej niż sesja Colaba, notebook ma komórkę konfiguracyjną z wersją skróconą i zapisuje wagi modelu, a opis jasno mówi, która wersja dała raportowane liczby.

## Harmonogram

| Etap | Co oddajecie | Moment |
|---|---|---|
| 1. Temat | Problem, zbiór danych z licencją, metryka główna i uzasadnienie jej wyboru, plan podziału na zbiory (pół strony) | po laboratorium 3 |
| 2. Stan pośredni | Notebook z wczytaniem danych, punktem odniesienia i pierwszą siecią z krzywymi uczenia | po laboratorium 5 |
| 3. Wersja końcowa | Notebook, prezentacja (do 7 minut plus pytania) i odpowiedzi na pytania o kod | ostatnie zajęcia |

Projekt kończy się prezentacją na zajęciach. Z pytaniami i stanem pośrednim prac możecie przyjść na konsultacje. Etapy 1 i 2 nie dostają osobnej oceny, ale ich brak obniża ocenę końcową (patrz kryteria). Służą temu, żeby wcześnie wychwycić problemy: zbyt duży zbiór na darmowy Colab, źle dobraną metrykę, przeciek danych między zbiorami.

## Wymagania

### Na ocenę 3,0

1. Problem i zbiór danych z listy tematów albo uzgodnione z prowadzącym. Opis danych: liczba przykładów, rozkład klas, przykładowe obrazy lub teksty, licencja i źródło.
2. Podział na zbiory treningowy, walidacyjny i testowy. Zbiór testowy jest użyty raz, na samym końcu, dla wybranego modelu. Jeżeli zbiór nie ma gotowej walidacji, wydzielacie ją z treningowego i piszecie jak.
3. Punkt odniesienia: wynik klasy większościowej oraz prosty model niebędący głęboką siecią (na przykład regresja logistyczna na TF-IDF dla tekstu albo na spłaszczonych pikselach lub cechach dla obrazów).
4. Co najmniej jedna sieć zaprojektowana samodzielnie i trenowana od zera (MLP, CNN, sieć rekurencyjna albo Conv1D), z krzywymi uczenia i komentarzem, czy model jest przeuczony lub niedouczony.
5. Metryka główna dobrana do problemu i uzasadniona (przy niezrównoważonych klasach sama dokładność nie wystarcza), macierz pomyłek i raport dla klas na zbiorze testowym.
6. Wnioski i ograniczenia: co działa, czego nie udało się osiągnąć i dlaczego.

### Na ocenę 4,0 (dodatkowo)

7. Model pretrenowany: transfer learning dla obrazów (ekstraktor cech i fine-tuning) albo fine-tuning modelu typu BERT dla tekstu, porównany z siecią trenowaną od zera.
8. Strojenie: co najmniej jedna technika regularyzacji (augmentacja, dropout, L2, early stopping) i systematyczne przeszukanie co najmniej dwóch hiperparametrów (KerasTuner albo zaplanowana siatka), z tabelą eksperymentów na zbiorze walidacyjnym.
9. Analiza błędów: przykłady błędnych predykcji z interpretacją, czego model nie rozumie. Dla obrazów mapa Grad-CAM dla kilku przypadków, dla tekstu przykłady i próba wskazania wzorca w pomyłkach.
10. Odtwarzalność: ustalone ziarno losowania, wypisane wersje bibliotek, notebook przechodzi "Uruchom wszystko" w świeżym środowisku.

### Na ocenę 5,0 (dodatkowo)

11. Niepewność wyniku: co najmniej trzy powtórzenia treningu z różnymi ziarnami albo przedział ufności metodą bootstrap na zbiorze testowym. Różnice między modelami odniesione do tej zmienności, a nie podane jako pojedyncze liczby.
12. Odporność: sprawdzenie modelu na danych spoza rozkładu treningowego (inna domena, własne zdjęcia, zaburzenia obrazu takie jak szum lub rozmycie, teksty z innego źródła) i omówienie spadku jakości.
13. Koszt: liczba parametrów, rozmiar pliku modelu, czas treningu i czas predykcji jednego przykładu dla porównywanych modeli oraz uzasadniony wybór modelu, który wdrożylibyście, gdyby liczył się koszt, a nie tylko jakość.
14. Krótka notatka dla odbiorcy nietechnicznego (po polsku, bez żargonu, do jednej strony): co model robi, jak często się myli i w jakich sytuacjach, czego nie wolno mu powierzać.

Ocena celująca (5,5) wymaga według sylabusa pisemnego uzasadnienia prowadzącego. Może ją dostać projekt, który wyraźnie wykracza poza wymagania na 5,0, na przykład własnym, starannie zebranym zbiorem danych z opisem procesu etykietowania, wdrożeniem modelu z pomiarem działania na prawdziwych danych albo nietrywialnym eksperymentem, który odpowiada na pytanie badawcze.

## Propozycje tematów

Zbiory z listy są dostępne publicznie jako pliki CSV albo Parquet na Hugging Face i dają się wczytać przez `pandas` bez logowania (sprawdzone w listopadzie 2026). Liczności i licencje podane niżej pochodzą z tych plików i ze stron źródłowych. W etapie 1 i tak sprawdzacie je sami. Zbioru PolEmo 2.0 i Oxford-IIIT Pet z laboratoriów nie wybieracie.

### Obrazy

Użytkowanie terenu ze zdjęć satelitarnych (EuroSAT). Zdjęcia Sentinel-2 w barwach RGB, 64 na 64 piksele, 10 klas (pola uprawne, las, autostrada, rzeka, zabudowa i inne). Plik `timm/eurosat-rgb`: 16 200 obrazów treningowych, 5 400 walidacyjnych i 5 400 testowych. Licencja MIT. Źródło: Helber i in., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification", 2019. Małe obrazy pozwalają trenować sieci od zera w rozsądnym czasie, a modele pretrenowane wymagają przeskalowania wejścia.

Choroby liści fasoli (Beans). Zdjęcia liści z Ugandy w trzech klasach: zdrowe, plamistość kątowa i rdza. Plik `AI-Lab-Makerere/beans`: 1 034 obrazy treningowe, 133 walidacyjne i 128 testowe, klasy prawie równoliczne. Licencja MIT. Zbiór jest mały, więc transfer learning jest tu koniecznością, a zbiór testowy tak mały, że różnica jednego obrazu to prawie punkt procentowy. Wymaganie 11 ma tu szczególne znaczenie.

Zapalenie płuc na zdjęciach rentgenowskich klatki piersiowej. Plik `hf-vision/chest-xray-pneumonia`: 5 216 obrazów treningowych (1 341 bez zmian, 3 875 z zapaleniem płuc), 16 walidacyjnych i 624 testowe. Licencja CC BY 4.0. Źródło: Kermany, Zhang, Goldbaum, "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, 2018. Walidację trzeba wydzielić z części treningowej, bo 16 obrazów to za mało. Wersja na Hugging Face nie zawiera identyfikatorów pacjentów, więc nie da się wykluczyć, że zdjęcia tej samej osoby trafią do treningu i walidacji. Trzeba to opisać jako ograniczenie. Model z tego projektu nie jest narzędziem diagnostycznym i notatka z wymagania 14 musi to mówić.

### Tekst po polsku

Wykrywanie cyberprzemocy w tweetach (CBD z benchmarku KLEJ, zadanie 6 konkursu PolEval 2019). Plik `allegro/klej-cbd`: 10 041 tweetów treningowych, z czego 851 (8,5%) oznaczono jako szkodliwe, i 1 000 testowych (134 szkodliwe). Brak gotowej walidacji. Licencja według strony KLEJ: BSD 3-Clause. Źródło: Ptaszynski, Pieciukiewicz, Dybała, "Results of the PolEval 2019 Shared Task 6", 2019. Silnie niezrównoważone klasy wymuszają przemyślany wybór metryki i progu decyzyjnego. Teksty zawierają wulgaryzmy i obraźliwe treści.

Ocena recenzji produktu (Allegro Reviews z benchmarku KLEJ). Plik `allegro/klej-allegro-reviews`: 9 577 recenzji treningowych, 1 002 walidacyjne i 1 006 testowych z oceną od 1 do 5 gwiazdek (najczęściej 5). Licencja według strony KLEJ: CC BY-SA 4.0. Źródło: Rybak i in., "KLEJ: Comprehensive Benchmark for Polish Language Understanding", ACL 2020. Problem można ująć jako klasyfikację albo regresję. Pomyłka o jedną gwiazdkę jest mniej poważna niż o cztery, a metryka powinna to uwzględniać (benchmark KLEJ używa ważonego średniego błędu bezwzględnego).

Typ nazwy własnej w zdaniu (NKJP-NER z benchmarku KLEJ). Plik `allegro/klej-nkjp-ner`: 15 794 zdania treningowe, 1 941 walidacyjnych i 2 058 testowych. Każde zdanie zawiera nazwy własne jednego typu albo żadnej, a zadanie polega na wskazaniu typu (osoba, miejsce, organizacja, obiekt geograficzny, czas, brak). Licencja według strony KLEJ: GNU GPL v3. Źródło: Narodowy Korpus Języka Polskiego (Przepiórkowski i in., PWN 2012). Rozkład klas w teście różni się od treningowego, co jest dobrym materiałem do wymagania 12.

Czy odpowiedź pasuje do pytania (DYK, "Czy wiesz?", z benchmarku KLEJ). Plik `allegro/klej-dyk`: 4 154 pary pytanie i odpowiedź w zbiorze treningowym (701 poprawnych) i 1 029 w testowym (174 poprawne). Licencja według strony KLEJ: CC BY-SA 3.0. Źródło: Marcińczuk i in., "Open dataset for development of Polish Question Answering systems", 2013. Wejściem są dwa teksty naraz, więc temat dobrze pokazuje przewagę modeli typu BERT nad siecią rekurencyjną. Niepoprawne odpowiedzi dobrano tak, by miały dużo wspólnych słów z pytaniem, co utrudnia zadanie metodom opartym na słowach.

### Temat własny

Zbiór z waszej pracy zawodowej, zainteresowań albo inny publiczny zbiór obrazów lub tekstów. Do etapu 1 dołączcie opis źródła, licencji i sposobu etykietowania. Danych firmowych i osobowych nie wolno używać bez zgody i anonimizacji, a do notebooka oddawanego na uczelnię nie trafiają w ogóle. Zbiór musi mieć co najmniej kilkaset przykładów z etykietami i zmieścić się w pamięci darmowego Colaba.

## Struktura notebooka

Minimalny układ (szkielet jest w `projekt_szablon.ipynb`):

1. Autor, problem, cel
2. Dane: źródło, licencja, opis, przykłady
3. Podział na zbiory i protokół ewaluacji, metryka główna
4. Punkt odniesienia
5. Sieć trenowana od zera
6. Model pretrenowany
7. Regularyzacja i strojenie hiperparametrów
8. Wybór modelu i jednorazowy test
9. Analiza błędów
10. Niepewność, odporność i koszt
11. Wnioski, ograniczenia, notatka dla odbiorcy nietechnicznego

Każda sekcja zawiera komentarz. Sam kod i wykresy bez słowa wyjaśnienia nie wystarczą.

## Kryteria oceny

| Kryterium | Waga | Co oceniamy |
|---|---|---|
| Problem, dane i protokół ewaluacji | 15% | Opis danych, licencja, poprawny podział, brak przecieku, uzasadniona metryka |
| Modele i trening | 25% | Punkt odniesienia, własna architektura, model pretrenowany, poprawność treningu, krzywe uczenia |
| Regularyzacja i strojenie | 15% | Świadomie dobrane techniki, systematyczne eksperymenty, tabela wyników na walidacji |
| Ewaluacja i analiza błędów | 20% | Jednorazowy test, macierz pomyłek, analiza przykładów, niepewność wyniku, odporność |
| Wnioski i odpowiedzialne użycie | 10% | Ograniczenia, koszt, notatka dla odbiorcy nietechnicznego, ryzyka zastosowania |
| Prezentacja i obrona | 15% | Jasność wypowiedzi, odpowiedzi na pytania o kod i decyzje |

Każde kryterium dostaje ocenę od 2 do 5, a ocenę końcową wyznacza średnia ważona, przeliczona według skali z sylabusa (od 3,00 do 3,25 dostateczna, od 3,26 do 3,75 dostateczna plus, od 3,76 do 4,25 dobra, od 4,26 do 4,75 dobra plus, od 4,76 do 5,0 bardzo dobra). Wymagania z poprzedniej sekcji wyznaczają górny pułap: projekt, który nie spełnia wymagań na 4,0, nie dostanie więcej niż 3,5, a niespełniający wymagań na 5,0 nie więcej niż 4,5. Brak etapu 1 albo 2 obniża ocenę końcową o pół stopnia za każdy brakujący etap.

## Format oddania

1. Notebook `.ipynb` z zapisanymi wynikami, wgrany w sposób wskazany przez prowadzącego, z linkiem do kopii w Colabie z uprawnieniem do odczytu.
2. Instalacja bibliotek spoza Colaba w pierwszej komórce.
3. Wagi najlepszego modelu nie są wymagane w oddaniu, ale notebook powinien umieć je odtworzyć.
4. Prezentacja: do 5 slajdów albo bezpośrednio notebook, do 7 minut, plus pytania.

## Praktyczne wskazówki

Zacznijcie od punktu odniesienia i najprostszej sieci, zanim sięgniecie po model pretrenowany. Bez punktu odniesienia nie wiadomo, czy 85% to dużo, czy mało. W części treningowej zbioru CBD 91,5% tweetów nie jest szkodliwych, więc model, który zawsze odpowiada "nieszkodliwy", ma tam 91,5% dokładności i nie umie nic.

Nie patrzcie na zbiór testowy przy wyborze modelu. Każda decyzja podjęta po obejrzeniu wyniku na teście (zmiana progu, liczby epok, architektury) sprawia, że wynik testu przestaje być uczciwym oszacowaniem.

Zapisujcie eksperymenty, także nieudane, w tabeli: konfiguracja, wynik walidacyjny, czas. Model, który wyszedł gorzej od punktu odniesienia, z wyjaśnieniem dlaczego, jest wartościowym wynikiem.

Darmowy Colab rozłącza sesję po pewnym czasie bezczynności i ma limit użycia GPU. Zapisujcie wagi po każdym dłuższym treningu (na przykład `ModelCheckpoint` na Dysk Google) i nie uruchamiajcie wszystkiego od nowa po każdej poprawce.
