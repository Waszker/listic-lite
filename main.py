import asyncio
from ai_agent.agent import run_agent


if __name__ == "__main__":
    example_inputs = [
        # "https://www.rozkoszny.pl/sernik-baskijski-ale-jeszcze-lepszy/",
        """
        Sernik baskijski (ale jeszcze lepszy)
tortownica 23 cm
180 g ulubionych ciasteczek, najlepiej digestives
3 łyżki masła
750 g serka śmietankowego typu Philadelphia (patrz Rady/porady)
250 g gęstego jogurtu greckiego, odsączonego z serwatki na sitku
1 ¼ szklanki (250 g) drobnoziarnistego cukru
6 dużych jajek „zerówek”
1 ½ szklanki (400 ml) śmietany kremówki 36%
1 płaska łyżeczka drobnej soli morskiej
1 łyżeczka ekstraktu waniliowego
2 łyżki cukru pudru (opcjonalnie)
        """,
        """

Składniki
4 porcje

    700 g karkówki wieprzowej (karczku)
    sól morska i świeżo zmielony czarny pieprz
    1 łyżeczka oliwy extra vergine lub 2 łyżeczki masła
    gałązka rozmarynu lub 2 łyżeczki suszonego
    2 ząbki czosnku
    2 łyżeczki sosu sojowego lub 1 łyżeczka worcestershire
    1/2 szklanki białego wytrawnego wina (jeśli nie mamy można użyć bulionu)
    100 g mrożonej marchewki mini lub zwykłej

""",
        """
    1 kg surowej białej kiełbasy
    2 cebule
    150 g parzonego / wędzonego boczku

Zalewa

    1/4 szklanki piwa (lub białego wina) lub 2 łyżki octu
    3 łyżki musztardy francuskiej (ziarnistej)
    2 łyżki sosu sojowego
    1 łyżeczka miodu
    1 łyżeczka suszonego majeranku
    1/4 łyżeczki zmielonego pieprzu
""",
        # "https://www.kwestiasmaku.com/przepis/pieczona-biala-kielbasa",
        # "https://www.kwestiasmaku.com/przepis/karkowka-pieczona-w-plastrach",
        # "https://www.kwestiasmaku.com/przepis/kurczak-w-sosie-curry/",
        # """
        # **Simple Pancake Recipe**
        # Ingredients:
        # - 1 szklanka mąki
        # - 2 łyżki cukru
        # - 2 łyżeczki proszku do pieczenia
        # - 1/2 łyżeczki soli
        # - 1 jajko
        # - 1 szklanka mleka
        # - 2 łyżki roztopionego masła
        # Instructions:
        # 1. Mix dry ingredients.
        # 2. Mix wet ingredients.
        # 3. Combine and cook.
        # """,
        # """
        # Lista zakupów:
        # Cebula 2 szt.
        # Czosnek 3 ząbki
        # Marchew 1kg
        # Papryka czerwona 1
        # Olej roślinny
        # """,
        # """
        # Do zrobienia zakupów potrzebujemy:
        # Pierś z kurczaka: 500g
        # Pałki kurczaka: 6szt
        # Cebula: pół małej cebuli
        # Czosnek: 3 główki
        # Marchew: sto gramów
        # Papryka czerwona: 120g
        # 3 marchewki,
        # 5 papryk żółtych,
        # karton mleka,
        # 100ml. mleka,
        # """,
        # "https://www.allrecipes.com/recipe/46822/indian-chicken-curry-ii/",
        # "https://www.indianhealthyrecipes.com/chicken-curry/",
        # "https://headbangerskitchen.com/indian-curry-chicken-curry/",
    ]

    asyncio.run(run_agent(example_inputs))
