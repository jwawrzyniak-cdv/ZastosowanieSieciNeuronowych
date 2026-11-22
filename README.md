# Zastosowanie sieci neuronowych: laboratoria w Google Colab

Materiały do laboratoriów z kursu "Zastosowanie sieci neuronowych" w module "Zaawansowana sztuczna inteligencja": sześć laboratoriów, projekt zaliczeniowy i nieobowiązkowe zadanie dodatkowe na platformie Databricks. Notebooki uruchamiacie w Google Colab z GPU. Kod korzysta z Keras 3.

## Plan zajęć

| Lab | Temat | Dane | Notebook |
|---|---|---|---|
| 1 | Perceptron wielowarstwowy, pętla treningowa | Fashion-MNIST | [lab01_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/lab01_mlp/lab01_szablon.ipynb) |
| 2 | Sieci konwolucyjne, przeuczenie, augmentacja, dropout, L2 | CIFAR-10 | [lab02_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/lab02_cnn/lab02_szablon.ipynb) |
| 3 | Transfer learning i fine-tuning | Oxford-IIIT Pet | [lab03_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/lab03_transfer_learning/lab03_szablon.ipynb) |
| 4 | Callbacki, optymalizatory, strojenie hiperparametrów, metryki, Grad-CAM | CIFAR-10 | [lab04_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/lab04_tuning_i_ewaluacja/lab04_szablon.ipynb) |
| 5 | Tekst: tokenizacja, embeddingi, LSTM, Conv1D | PolEmo 2.0 | [lab05_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/lab05_tekst_rnn/lab05_szablon.ipynb) |
| 6 | Mechanizm uwagi, transformery, fine-tuning HerBERT | PolEmo 2.0 | [lab06_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/lab06_transformery/lab06_szablon.ipynb) |
| 7 | Projekt zaliczeniowy: konsultacje i prezentacje | do wyboru | [projekt_szablon](https://colab.research.google.com/github/jwawrzyniak-cdv/ZastosowanieSieciNeuronowych/blob/main/projekt_zaliczeniowy/projekt_szablon.ipynb) |

Na końcu każdej instrukcji jest sekcja "Zadania rozszerzone" z zadaniami wykraczającymi poza podstawowy zakres laboratorium. Prowadzący powie, czy są w waszej grupie obowiązkowe.

## Jak pracować z materiałami

Przed zajęciami przeczytajcie `labNN_wprowadzenie.md`, czyli materiał wprowadzeniowy z pojęciami i ilustracjami potrzebnymi do zadań, a potem `labNN_instrukcja.md`. Na zajęciach pracujecie w notebooku `labNN_szablon.ipynb`: części pomocnicze (wczytanie danych, funkcje do rysowania) są gotowe, a miejsca oznaczone `# TODO` uzupełniacie sami. Odpowiedzi na pytania wpisujecie w komórkach "Odpowiedzi" w notebooku.

Notebook otworzycie w Colabie z linku w tabeli powyżej. Od razu zapiszcie kopię na Dysku Google (Plik, Zapisz kopię na Dysku), bo zmiany w notebooku otwartym z GitHuba giną po zamknięciu karty. Przed uruchomieniem włączcie GPU: Środowisko wykonawcze, Zmień typ środowiska wykonawczego, T4 GPU. Darmowy Colab ma limit czasu pracy na GPU i rozłącza nieaktywne sesje, więc zapisujcie wagi modeli po dłuższych treningach.

Kilka rzeczy, o których warto wiedzieć wcześniej:

1. Wszystkie dane i modele pobiera kod z internetu (Keras, Hugging Face). Laboratorium 3 pobiera około 790 MB zdjęć, a laboratorium 6 model HerBERT o rozmiarze około 650 MB, co chwilę trwa.
2. Laboratorium 4 instaluje biblioteki `keras-tuner` i `tensorboard`, bo pierwszej z nich nie ma w Colabie domyślnie. Do części z Grad-CAM przyda się plik `lab03_pets.keras` zapisany w laboratorium 3; bez niego notebook użyje waszej własnej sieci. Instalacja jest w pierwszej komórce notebooka.
3. Laboratorium 6 uruchamia Keras na backendzie PyTorch zamiast TensorFlow, bo obecna wersja biblioteki transformers, z której pochodzi model HerBERT, nie obsługuje już TensorFlow. Backend ustawia się przed pierwszym `import keras`, więc notebook trzeba uruchomić w świeżej sesji (Środowisko wykonawcze, Uruchom ponownie sesję), a nie po innym notebooku.

## Zasady zaliczenia

Według sylabusa podstawą zaliczenia kursu jest zadanie realizowane w trakcie zajęć lub na ich końcu. Zadaniami w trakcie zajęć są laboratoria, a zadaniem na końcu projekt zaliczeniowy. Ocena z kursu stanowi 50% oceny z modułu "Zaawansowana sztuczna inteligencja". Sposób połączenia oceny z laboratoriów i z projektu poda prowadzący na pierwszych zajęciach.

Projekt wykonujecie indywidualnie. Wymagania, kryteria oceny i propozycje tematów są w pliku `projekt_zaliczeniowy/projekt_opis.md`, a szkielet notebooka w `projekt_zaliczeniowy/projekt_szablon.ipynb`. Projekt ma trzy etapy: temat po laboratorium 3, stan pośredni po laboratorium 5 i wersja końcowa z prezentacją na ostatnich zajęciach.

## Zadanie dodatkowe: Databricks Free Edition

W katalogu `zadanie_dodatkowe_databricks` jest nieobowiązkowe zadanie, w którym sieć CNN wytrenowana w Colabie przechodzi przez proces wdrożenia: zapis przebiegu w MLflow, rejestrację w Unity Catalog i uruchomienie endpointu REST na bezpłatnej platformie Databricks Free Edition. Wynik co najmniej 50 punktów na 100 podnosi średnią z laboratoriów o pół stopnia, ale nie powyżej 5,0. Szczegóły, w tym zakładanie konta i ograniczenia platformy, są w `dbx_instrukcja.md`.

## Literatura

Literatura podstawowa:

1. I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016. Wersja online: https://www.deeplearningbook.org/

Literatura uzupełniająca:

1. J. Krohn, G. Beyleveld, A. Bassens, Uczenie głębokie i sztuczna inteligencja. Interaktywny przewodnik ilustrowany, Helion, Gliwice 2021.
2. F. Chollet, Deep Learning with Python, Second Edition, Manning, 2021.

Dokumentacja Keras 3: https://keras.io/

Źródła do poszczególnych tematów (rozdziały książek, artykuły o zbiorach danych i modelach) są wymienione na końcu plików `labNN_instrukcja.md`.
