import asyncio
from ai_agent.agent import run_agent


if __name__ == "__main__":
    example_inputs = [
        "https://www.kwestiasmaku.com/przepis/kurczak-w-sosie-curry",
        """
        **Simple Pancake Recipe**

        Ingredients:
        - 1 szklanka mąki
        - 2 łyżki cukru
        - 2 łyżeczki proszku do pieczenia
        - 1/2 łyżeczki soli
        - 1 jajko
        - 1 szklanka mleka
        - 2 łyżki roztopionego masła

        Instructions:
        1. Mix dry ingredients.
        2. Mix wet ingredients.
        3. Combine and cook.
        """,
        """
        Lista zakupów:
        Cebula 2 szt.
        Czosnek 3 ząbki
        Marchew 1kg
        Papryka czerwona 1
        Olej roślinny
        """,
        """
        Do zrobienia zakupów potrzebujemy:
        Pierś z kurczaka: 500g
        Pałki kurczaka: 6szt
        Cebula: pół małej cebuli
        Czosnek: 3 główki
        Marchew: sto gramów
        Papryka czerwona: 120g
        3 marchewki,
        5 papryk żółtych,
        karton mleka,
        100ml. mleka,
        """,
        "https://www.allrecipes.com/recipe/46822/indian-chicken-curry-ii/",
        "https://www.indianhealthyrecipes.com/chicken-curry/",
        "https://headbangerskitchen.com/indian-curry-chicken-curry/",
    ]

    asyncio.run(run_agent(example_inputs))
